#include <JuceHeader.h>

class SwellProcessor : public juce::AudioProcessor
{
public:
    enum State { idle, armed, recording, captured };
    enum Mode { none, playDry, playSwell };

    juce::AudioProcessorValueTreeState params;
    std::atomic<int> state { idle }, mode { none };
    std::atomic<double> bpm { 120.0 };
    std::atomic<int> beatsPerBar { 4 };
    juce::AudioBuffer<float> capture;   // written by audio thread only while armed/recording
    double rate = 44100.0, captureRate = 44100.0;

    SwellProcessor()
        : AudioProcessor (BusesProperties().withInput ("In", juce::AudioChannelSet::stereo(), true)
                                           .withOutput ("Out", juce::AudioChannelSet::stereo(), true)),
          params (*this, nullptr, "params", layout()) {}

    static juce::AudioProcessorValueTreeState::ParameterLayout layout()
    {
        juce::AudioProcessorValueTreeState::ParameterLayout l;
        auto knob = [&l] (const char* id, const char* name, float lo, float hi, float def)
        {
            l.add (std::make_unique<juce::AudioParameterFloat> (juce::ParameterID { id, 1 }, name,
                       juce::NormalisableRange<float> (lo, hi, 0.01f), def));
        };
        l.add (std::make_unique<juce::AudioParameterChoice> (juce::ParameterID { "length", 1 }, "Length",
                   juce::StringArray { "1/4 bar", "1/2 bar", "1 bar", "2 bars" }, 2));
        knob ("size", "Size", 0.0f, 1.0f, 0.5f);
        knob ("tone", "Tone", 0.0f, 1.0f, 0.5f);
        knob ("width", "Width", 0.0f, 1.0f, 0.6f);
        knob ("trem", "Tremolo", 0.0f, 1.0f, 0.0f);
        l.add (std::make_unique<juce::AudioParameterChoice> (juce::ParameterID { "rate", 1 }, "Tremolo Rate",
                   juce::StringArray { "1/8", "1/16", "1/32" }, 1));
        l.add (std::make_unique<juce::AudioParameterBool> (juce::ParameterID { "lowcut", 1 }, "Low Cut", false));
        knob ("output", "Output dB", -24.0f, 0.0f, -1.0f);
        return l;
    }

    float value (const char* id) const { return params.getRawParameterValue (id)->load(); }

    void prepareToPlay (double sr, int) override
    {
        const juce::SpinLock::ScopedLockType sl (lock);
        mode = none;
        rate = sr;
        if (state != captured)
        {
            captureRate = sr;
            capture.setSize (2, (int) (sr * 6.0));
            state = idle;
        }
    }
    void releaseResources() override {}

    bool isBusesLayoutSupported (const BusesLayout& b) const override
    {
        return b.getMainOutputChannelSet() == juce::AudioChannelSet::stereo()
            && b.getMainInputChannelSet() == juce::AudioChannelSet::stereo();
    }

    void arm()    { mode = none; position = 0; state = armed; }
    void finish() { if (state == recording) state = captured; else if (state == armed) state = idle; }
    int capturedLength() const { return position.load(); }

    void processBlock (juce::AudioBuffer<float>& buffer, juce::MidiBuffer&) override
    {
        juce::ScopedNoDenormals noDenormals;
        if (auto* head = getPlayHead())
            if (auto pos = head->getPosition())
            {
                if (auto b = pos->getBpm()) bpm = *b;
                if (auto t = pos->getTimeSignature()) beatsPerBar = t->numerator;
            }

        const int n = buffer.getNumSamples();
        int start = 0;
        if (state == armed)
            for (; start < n; ++start)
                if (std::abs (buffer.getSample (0, start)) > 0.01f || std::abs (buffer.getSample (1, start)) > 0.01f)
                {
                    int expected = armed;
                    state.compare_exchange_strong (expected, recording);
                    break;
                }

        if (state == recording)
        {
            const int pos = position.load();
            const int count = juce::jmin (n - start, capture.getNumSamples() - pos);
            for (int c = 0; c < 2; ++c)
                capture.copyFrom (c, pos, buffer, c, start, count);
            position = pos + count;
            if (pos + count >= capture.getNumSamples()) state = captured;
        }

        const juce::SpinLock::ScopedTryLockType sl (lock);
        if (sl.isLocked() && mode != none)
        {
            const auto& src = mode == playDry ? dry : swell;
            const int count = juce::jmin (n, src.getNumSamples() - playPos);
            buffer.clear();
            for (int c = 0; c < 2 && count > 0; ++c)
                buffer.copyFrom (c, 0, src, c, playPos, count);
            playPos += juce::jmax (0, count);
            if (playPos >= src.getNumSamples()) mode = none;
        }
    }

    void play (Mode m)
    {
        const juce::SpinLock::ScopedLockType sl (lock);
        playPos = 0;
        mode = m;
    }

    // Reverb the opening of the capture, then reverse the tail so it rises into the onset.
    bool render()
    {
        if (state != captured || position.load() < 64) return false;
        const double sr = captureRate;
        const double bars[] = { 0.25, 0.5, 1.0, 2.0 };
        const double tempo = juce::jlimit (20.0, 400.0, bpm.load());
        const int n = (int) (sr * 60.0 / tempo * beatsPerBar.load() * bars[(int) value ("length")]);
        const int headLen = juce::jmin (position.load(), (int) (sr * 0.5));
        const int irLen = (int) (sr * (0.4 + value ("size") * 5.0));
        int order = 1;
        while ((1 << order) < juce::jmax (n, headLen + irLen)) ++order;
        const int fftSize = 1 << order;
        juce::dsp::FFT fft (order);
        juce::AudioBuffer<float> out (2, n);
        const float width = value ("width");
        juce::Random r1 (1234), r2 (98765);
        std::vector<float> left ((size_t) irLen), right ((size_t) irLen);
        for (int i = 0; i < irLen; ++i)
        {
            const float env = std::exp (-6.9f * (float) i / (float) irLen);
            const float m = r1.nextFloat() * 2.0f - 1.0f, s = r2.nextFloat() * 2.0f - 1.0f;
            left[(size_t) i]  = (m + s * width) * env;
            right[(size_t) i] = (m - s * width) * env;
        }
        const double cutoff = juce::jmin (800.0 * std::pow (22.0, (double) value ("tone")), sr * 0.45);
        for (int c = 0; c < 2; ++c)
        {
            std::vector<float> a ((size_t) fftSize * 2, 0.0f), b ((size_t) fftSize * 2, 0.0f);
            const int fade = juce::jmin (headLen, (int) (sr * 0.02));
            for (int i = 0; i < headLen; ++i)
                a[(size_t) i] = capture.getSample (c, i) * juce::jmin (1.0f, (float) (headLen - 1 - i) / (float) juce::jmax (1, fade));
            const auto& ir = c == 0 ? left : right;
            std::copy (ir.begin(), ir.end(), b.begin());
            fft.performRealOnlyForwardTransform (a.data());
            fft.performRealOnlyForwardTransform (b.data());
            for (size_t k = 0; k < (size_t) fftSize; ++k)
            {
                const std::complex<float> x (a[2 * k], a[2 * k + 1]), y (b[2 * k], b[2 * k + 1]);
                const auto z = x * y;
                a[2 * k] = z.real();
                a[2 * k + 1] = z.imag();
            }
            fft.performRealOnlyInverseTransform (a.data());
            juce::IIRFilter lp, hp;
            lp.setCoefficients (juce::IIRCoefficients::makeLowPass (sr, cutoff));
            hp.setCoefficients (juce::IIRCoefficients::makeHighPass (sr, value ("lowcut") > 0.5f ? 200.0 : 20.0));
            lp.processSamples (a.data(), n);
            hp.processSamples (a.data(), n);
            std::reverse_copy (a.begin(), a.begin() + n, out.getWritePointer (c));
        }
        const double hz = tempo / 60.0 * (2 << (int) value ("rate"));
        const float depth = value ("trem");
        const int edge = juce::jmax (1, (int) (sr * 0.005));
        for (int i = 0; i < n; ++i)
        {
            const float t = 1.0f - depth * (float) (0.5 - 0.5 * std::cos (juce::MathConstants<double>::twoPi * hz * i / sr));
            const float g = t * juce::jmin (1.0f, (float) i / (float) edge, (float) (n - 1 - i) / (float) edge);
            for (int c = 0; c < 2; ++c) out.getWritePointer (c)[i] *= g;
        }
        const float peak = out.getMagnitude (0, n);
        if (peak <= 0.0f) return false;
        out.applyGain (juce::Decibels::decibelsToGain (value ("output")) / peak);

        juce::AudioBuffer<float> raw (2, position.load());
        for (int c = 0; c < 2; ++c) raw.copyFrom (c, 0, capture, c, 0, raw.getNumSamples());

        const juce::SpinLock::ScopedLockType sl (lock);
        mode = none;
        swell = std::move (out);
        dry = std::move (raw);
        return true;
    }

    juce::File exportWav()
    {
        auto dir = juce::File::getSpecialLocation (juce::File::userMusicDirectory).getChildFile ("Swell Lab");
        if (! dir.createDirectory()) return {};
        auto file = dir.getNonexistentChildFile ("swell_" + juce::Time::getCurrentTime().formatted ("%Y%m%d_%H%M%S"), ".wav");
        std::unique_ptr<juce::FileOutputStream> stream (file.createOutputStream());
        if (stream == nullptr) return {};
        juce::WavAudioFormat format;
        std::unique_ptr<juce::AudioFormatWriter> writer (format.createWriterFor (stream.get(), captureRate, 2, 24, {}, 0));
        if (writer == nullptr) return {};
        stream.release();
        if (! writer->writeFromAudioSampleBuffer (swell, 0, swell.getNumSamples())) return {};
        return file;
    }

    const juce::AudioBuffer<float>& result() const { return swell; }

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }
    const juce::String getName() const override { return "Swell Lab"; }
    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }
    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram (int) override {}
    const juce::String getProgramName (int) override { return {}; }
    void changeProgramName (int, const juce::String&) override {}
    void getStateInformation (juce::MemoryBlock& d) override
    {
        if (auto xml = params.copyState().createXml()) copyXmlToBinary (*xml, d);
    }
    void setStateInformation (const void* d, int size) override
    {
        if (auto xml = getXmlFromBinary (d, size)) params.replaceState (juce::ValueTree::fromXml (*xml));
    }

private:
    juce::SpinLock lock;
    juce::AudioBuffer<float> dry, swell;
    std::atomic<int> position { 0 };
    int playPos = 0;
};

class DragButton : public juce::TextButton
{
public:
    std::function<void()> onDragStart;
    DragButton() : juce::TextButton ("DRAG TO FL") {}
    void mouseDown (const juce::MouseEvent& e) override { started = false; juce::TextButton::mouseDown (e); }
    void mouseDrag (const juce::MouseEvent& e) override
    {
        if (isEnabled() && ! started && e.getDistanceFromDragStart() > 6)
        {
            started = true;
            if (onDragStart) onDragStart();
        }
    }
private:
    bool started = false;
};

class SwellEditor : public juce::AudioProcessorEditor, private juce::Timer
{
public:
    explicit SwellEditor (SwellProcessor& p) : AudioProcessorEditor (p), proc (p)
    {
        for (auto* b : { (juce::Button*) &captureButton, (juce::Button*) &playButton,
                         (juce::Button*) &previewButton, (juce::Button*) &dragButton, (juce::Button*) &lowCut })
            addAndMakeVisible (b);
        captureButton.onClick = [this]
        {
            if (proc.state == SwellProcessor::armed || proc.state == SwellProcessor::recording) proc.finish();
            else { ready = false; dirty = true; proc.arm(); }
        };
        playButton.onClick = [this] { proc.play (SwellProcessor::playDry); };
        previewButton.onClick = [this] { proc.play (SwellProcessor::playSwell); };
        dragButton.onDragStart = [this]
        {
            if (dirty) update();
            if (! ready) return;
            const auto file = proc.exportWav();
            if (file.existsAsFile())
                juce::DragAndDropContainer::performExternalDragDropOfFiles ({ file.getFullPathName() }, false, this);
            else
                message = "Could not write the WAV to your Music folder.";
        };

        setupBox (length, "length", lengthAttach);
        setupBox (tremRate, "rate", rateAttach);
        const char* ids[] = { "size", "tone", "width", "trem", "output" };
        const char* names[] = { "SIZE", "TONE", "WIDTH", "TREMOLO", "OUTPUT dB" };
        for (int i = 0; i < 5; ++i)
        {
            knobs[i].setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
            knobs[i].setTextBoxStyle (juce::Slider::TextBoxBelow, false, 70, 18);
            knobs[i].setColour (juce::Slider::rotarySliderFillColourId, accent);
            addAndMakeVisible (knobs[i]);
            labels[i].setText (names[i], juce::dontSendNotification);
            labels[i].setJustificationType (juce::Justification::centred);
            addAndMakeVisible (labels[i]);
            knobAttach[i] = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (proc.params, ids[i], knobs[i]);
            knobs[i].onValueChange = [this] { dirty = true; };
        }
        lowCutAttach = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (proc.params, "lowcut", lowCut);
        lowCut.onStateChange = [this]
        {
            if (lowCut.getToggleState() != lastLowCut) { lastLowCut = lowCut.getToggleState(); dirty = true; }
        };
        lastLowCut = lowCut.getToggleState();
        setSize (620, 400);
        startTimerHz (10);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (juce::Colour (0xff10201e));
        g.setColour (juce::Colours::white);
        g.setFont (24.0f);
        g.drawText ("SWELL LAB", 20, 10, 300, 30, juce::Justification::centredLeft);
        g.setFont (14.0f);
        g.drawText (juce::String (proc.bpm.load(), 1) + " BPM   " + juce::String (proc.beatsPerBar.load()) + " beats/bar",
                    300, 10, 300, 30, juce::Justification::centredRight);
        const juce::Rectangle<int> screen (20, 50, 580, 120);
        g.setColour (juce::Colour (0xff1b3330));
        g.fillRoundedRectangle (screen.toFloat(), 8.0f);
        g.setColour (accent);
        const auto& buf = proc.result();
        if (ready && buf.getNumSamples() > 0)
        {
            const int step = juce::jmax (1, buf.getNumSamples() / 560);
            for (int x = 0; x < 560 && x * step < buf.getNumSamples(); ++x)
            {
                const float m = buf.getMagnitude (0, x * step, juce::jmin (step, buf.getNumSamples() - x * step));
                g.fillRect ((float) (30 + x), 110.0f - m * 55.0f, 1.0f, m * 110.0f + 1.0f);
            }
        }
        else
            g.drawText (message, screen, juce::Justification::centred);
    }

    void resized() override
    {
        captureButton.setBounds (20, 182, 139, 38);
        playButton.setBounds (167, 182, 139, 38);
        previewButton.setBounds (314, 182, 139, 38);
        dragButton.setBounds (461, 182, 139, 38);
        for (int i = 0; i < 5; ++i)
        {
            labels[i].setBounds (20 + i * 116, 235, 116, 20);
            knobs[i].setBounds (20 + i * 116, 255, 116, 90);
        }
        length.setBounds (30, 358, 130, 26);
        tremRate.setBounds (378, 358, 96, 26);
        lowCut.setBounds (190, 358, 120, 26);
    }

private:
    void setupBox (juce::ComboBox& box, const char* id,
                   std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment>& attach)
    {
        box.addItemList (proc.params.getParameter (id)->getAllValueStrings(), 1);
        addAndMakeVisible (box);
        attach = std::make_unique<juce::AudioProcessorValueTreeState::ComboBoxAttachment> (proc.params, id, box);
        box.onChange = [this] { dirty = true; };
    }

    void update()
    {
        ready = proc.render();
        dirty = false;
    }

    void timerCallback() override
    {
        const int s = proc.state;
        // stop capturing after one second of audio; only the first half second is used
        if (s == SwellProcessor::recording && proc.capturedLength() > (int) proc.captureRate)
            proc.finish();
        const double tempo = proc.bpm.load() * 100.0 + proc.beatsPerBar.load();
        if (std::abs (tempo - lastTempo) > 0.001) { lastTempo = tempo; dirty = true; }
        if (proc.state == SwellProcessor::captured && dirty && ! juce::ModifierKeys::currentModifiers.isAnyMouseButtonDown())
            update();
        if (s == SwellProcessor::armed) message = "Waiting for sound... press play in FL";
        else if (s == SwellProcessor::recording) message = "Capturing...";
        else if (s == SwellProcessor::idle) message = "Press CAPTURE, then play the sound";
        captureButton.setButtonText (s == SwellProcessor::armed || s == SwellProcessor::recording ? "STOP" : "CAPTURE");
        playButton.setEnabled (ready);
        previewButton.setEnabled (ready);
        dragButton.setEnabled (ready);
        repaint();
    }

    SwellProcessor& proc;
    const juce::Colour accent { 0xff4fd1b5 };
    juce::TextButton captureButton { "CAPTURE" }, playButton { "PLAY" }, previewButton { "PREVIEW" };
    DragButton dragButton;
    juce::ToggleButton lowCut { "Low cut" };
    juce::ComboBox length, tremRate;
    juce::Slider knobs[5];
    juce::Label labels[5];
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> knobAttach[5];
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> lengthAttach, rateAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lowCutAttach;
    juce::String message { "Press CAPTURE, then play the sound" };
    bool ready = false, dirty = true, lastLowCut = false;
    double lastTempo = 0.0;
};

juce::AudioProcessorEditor* SwellProcessor::createEditor() { return new SwellEditor (*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SwellProcessor(); }
