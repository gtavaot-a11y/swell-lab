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

namespace theme
{
    const juce::Colour back { 0xff0d1117 }, panel { 0xff171d27 }, line { 0xff2a3342 },
                       text { 0xffeef2f7 }, dim { 0xff8591a5 }, amber { 0xffffb020 }, coral { 0xffff5d73 };
    inline juce::Font font (float size, bool bold = false)
    {
        return juce::Font (juce::FontOptions (size, bold ? juce::Font::bold : juce::Font::plain));
    }
}

class SwellLook : public juce::LookAndFeel_V4
{
public:
    SwellLook()
    {
        setColour (juce::Slider::textBoxTextColourId, theme::text);
        setColour (juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
        setColour (juce::ComboBox::textColourId, theme::text);
        setColour (juce::PopupMenu::backgroundColourId, theme::panel);
        setColour (juce::PopupMenu::textColourId, theme::text);
        setColour (juce::PopupMenu::highlightedBackgroundColourId, theme::amber);
        setColour (juce::PopupMenu::highlightedTextColourId, theme::back);
        setColour (juce::TextButton::textColourOffId, theme::text);
    }

    void drawRotarySlider (juce::Graphics& g, int x, int y, int w, int h, float pos,
                           float start, float end, juce::Slider& slider) override
    {
        const auto area = juce::Rectangle<int> (x, y, w, h).toFloat();
        const float r = juce::jmin (area.getWidth(), area.getHeight()) * 0.5f - 5.0f;
        const auto c = area.getCentre();
        const float angle = start + pos * (end - start);
        const juce::PathStrokeType stroke (5.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded);

        juce::Path track, fill;
        track.addCentredArc (c.x, c.y, r, r, 0.0f, start, end, true);
        fill.addCentredArc (c.x, c.y, r, r, 0.0f, start, angle, true);
        g.setColour (theme::line);
        g.strokePath (track, stroke);
        g.setGradientFill (juce::ColourGradient (theme::amber, c.x - r, c.y, theme::coral, c.x + r, c.y, false));
        g.strokePath (fill, stroke);

        const float body = r - 8.0f;
        g.setGradientFill (juce::ColourGradient (juce::Colour (0xff323c4e), c.x, c.y - body,
                                                 juce::Colour (0xff151a23), c.x, c.y + body, false));
        g.fillEllipse (c.x - body, c.y - body, body * 2.0f, body * 2.0f);
        g.setColour (slider.isMouseOverOrDragging() ? theme::amber : theme::text);
        g.drawLine ({ c.getPointOnCircumference (body * 0.35f, angle),
                      c.getPointOnCircumference (body * 0.85f, angle) }, 3.0f);
    }

    void drawButtonBackground (juce::Graphics& g, juce::Button& button, const juce::Colour&,
                               bool over, bool down) override
    {
        const auto area = button.getLocalBounds().toFloat().reduced (1.0f);
        const bool primary = (bool) button.getProperties()["primary"];
        const float alpha = button.isEnabled() ? 1.0f : 0.4f;
        if (primary)
            g.setGradientFill (juce::ColourGradient (theme::amber.withAlpha (alpha), area.getX(), 0.0f,
                                                     theme::coral.withAlpha (alpha), area.getRight(), 0.0f, false));
        else
            g.setColour (theme::panel.brighter (down ? 0.25f : over ? 0.12f : 0.0f).withMultipliedAlpha (alpha));
        g.fillRoundedRectangle (area, 10.0f);
        g.setColour (primary && over ? theme::text : theme::line);
        g.drawRoundedRectangle (area, 10.0f, 1.0f);
    }

    juce::Font getTextButtonFont (juce::TextButton&, int) override { return theme::font (15.0f, true); }
    juce::Font getComboBoxFont (juce::ComboBox&) override { return theme::font (14.0f); }

    void drawComboBox (juce::Graphics& g, int w, int h, bool, int, int, int, int, juce::ComboBox&) override
    {
        const juce::Rectangle<float> area (0.5f, 0.5f, (float) w - 1.0f, (float) h - 1.0f);
        g.setColour (theme::back);
        g.fillRoundedRectangle (area, 7.0f);
        g.setColour (theme::line);
        g.drawRoundedRectangle (area, 7.0f, 1.0f);
        juce::Path arrow;
        const float x = (float) w - 17.0f, y = (float) h * 0.5f;
        arrow.addTriangle (x - 4.0f, y - 2.0f, x + 4.0f, y - 2.0f, x, y + 3.0f);
        g.setColour (theme::amber);
        g.fillPath (arrow);
    }

    void drawToggleButton (juce::Graphics& g, juce::ToggleButton& button, bool, bool) override
    {
        const bool on = button.getToggleState();
        const juce::Rectangle<float> pill (2.0f, (float) button.getHeight() * 0.5f - 11.0f, 42.0f, 22.0f);
        g.setColour (on ? theme::amber : theme::line);
        g.fillRoundedRectangle (pill, 11.0f);
        g.setColour (theme::text);
        g.fillEllipse (pill.getX() + (on ? 23.0f : 3.0f), pill.getY() + 3.0f, 16.0f, 16.0f);
        g.setFont (theme::font (14.0f));
        g.drawText (on ? "On" : "Off", 52, 0, button.getWidth() - 52, button.getHeight(),
                    juce::Justification::centredLeft);
    }
};

class DragButton : public juce::TextButton
{
public:
    std::function<void()> onDragStart;
    DragButton() : juce::TextButton ("DRAG TO FL")
    {
        setMouseCursor (juce::MouseCursor::DraggingHandCursor);
    }
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

// Everything is laid out at 760 x 500 and scaled to fit the window.
class SwellPanel : public juce::Component, private juce::Timer
{
public:
    explicit SwellPanel (SwellProcessor& p) : proc (p)
    {
        for (auto* b : { (juce::Button*) &captureButton, (juce::Button*) &playButton,
                         (juce::Button*) &previewButton, (juce::Button*) &dragButton, (juce::Button*) &lowCut })
            addAndMakeVisible (b);
        captureButton.getProperties().set ("primary", true);
        captureButton.setColour (juce::TextButton::textColourOffId, theme::back);
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
            {
                ready = false;
                message = "Could not write the WAV to your Music folder.";
            }
        };

        setupBox (length, "length", lengthAttach);
        setupBox (tremRate, "rate", rateAttach);
        const char* ids[] = { "size", "tone", "width", "trem", "output" };
        for (int i = 0; i < 5; ++i)
        {
            knobs[i].setSliderStyle (juce::Slider::RotaryHorizontalVerticalDrag);
            knobs[i].setTextBoxStyle (juce::Slider::TextBoxBelow, false, 80, 20);
            addAndMakeVisible (knobs[i]);
            knobAttach[i] = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment> (proc.params, ids[i], knobs[i]);
            if (i < 4)
            {
                knobs[i].textFromValueFunction = [] (double v) { return juce::String (juce::roundToInt (v * 100.0)) + "%"; };
                knobs[i].valueFromTextFunction = [] (const juce::String& t) { return t.getDoubleValue() / 100.0; };
            }
            else
                knobs[i].textFromValueFunction = [] (double v) { return juce::String (v, 1) + " dB"; };
            knobs[i].updateText();
            knobs[i].onValueChange = [this] { dirty = true; };
        }
        lowCutAttach = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment> (proc.params, "lowcut", lowCut);
        lastLowCut = lowCut.getToggleState();
        setSize (760, 500);
        timerCallback();
        startTimerHz (15);
    }

    void paint (juce::Graphics& g) override
    {
        g.fillAll (theme::back);

        g.setFont (theme::font (28.0f, true));
        g.setColour (theme::text);
        g.drawText ("SWELL", 24, 16, 110, 32, juce::Justification::centredLeft);
        g.setColour (theme::amber);
        g.drawText ("LAB", 132, 16, 80, 32, juce::Justification::centredLeft);
        g.setFont (theme::font (13.0f));
        g.setColour (theme::dim);
        g.drawText ("Reverse reverb swells", 24, 46, 220, 18, juce::Justification::centredLeft);

        chip (g, { 548, 22, 96, 32 }, juce::String (proc.bpm.load(), 1) + " BPM");
        chip (g, { 652, 22, 84, 32 }, juce::String (proc.beatsPerBar.load()) + " beats");

        const juce::Rectangle<float> screen (24.0f, 76.0f, 712.0f, 170.0f);
        g.setColour (theme::panel);
        g.fillRoundedRectangle (screen, 12.0f);
        g.setColour (theme::line);
        g.drawRoundedRectangle (screen, 12.0f, 1.0f);

        const auto& buf = proc.result();
        if (ready && buf.getNumSamples() > 0)
        {
            g.setGradientFill (juce::ColourGradient (theme::amber.withAlpha (0.5f), 40.0f, 0.0f,
                                                     theme::coral, 720.0f, 0.0f, false));
            const int total = buf.getNumSamples();
            for (int x = 0; x < 226; ++x)
            {
                const int from = (int) ((juce::int64) x * total / 226);
                const int count = juce::jmax (1, (int) ((juce::int64) (x + 1) * total / 226) - from);
                const float m = juce::jmin (1.0f, buf.getMagnitude (from, juce::jmin (count, total - from)));
                g.fillRoundedRectangle (42.0f + (float) x * 3.0f, 161.0f - m * 60.0f, 2.0f, m * 120.0f + 2.0f, 1.0f);
            }
            g.setFont (theme::font (12.0f));
            g.setColour (theme::dim);
            g.drawText (juce::String (total / proc.captureRate, 2) + " s", 40, 224, 100, 18, juce::Justification::centredLeft);
            g.drawText ("your sound starts here >", 520, 224, 200, 18, juce::Justification::centredRight);
        }
        else
        {
            const bool busy = proc.state == SwellProcessor::armed || proc.state == SwellProcessor::recording;
            g.setColour (busy ? theme::coral : theme::dim);
            g.setFont (theme::font (17.0f, busy));
            g.drawText (message, screen.toNearestInt(), juce::Justification::centred);
        }

        g.setColour (theme::panel);
        g.fillRoundedRectangle (24.0f, 316.0f, 712.0f, 164.0f, 12.0f);
        const char* names[] = { "SIZE", "TONE", "WIDTH", "TREMOLO", "OUTPUT" };
        g.setColour (theme::dim);
        g.setFont (theme::font (12.0f, true));
        for (int i = 0; i < 5; ++i)
            g.drawText (names[i], 34 + i * 112, 328, 112, 16, juce::Justification::centred);
        g.drawText ("LENGTH", 606, 328, 112, 16, juce::Justification::centredLeft);
        g.drawText ("TREMOLO RATE", 606, 382, 112, 16, juce::Justification::centredLeft);
        g.drawText ("LOW CUT", 606, 434, 112, 16, juce::Justification::centredLeft);
    }

    void resized() override
    {
        captureButton.setBounds (24, 258, 172, 46);
        playButton.setBounds (204, 258, 172, 46);
        previewButton.setBounds (384, 258, 172, 46);
        dragButton.setBounds (564, 258, 172, 46);
        for (int i = 0; i < 5; ++i)
            knobs[i].setBounds (34 + i * 112, 350, 112, 118);
        length.setBounds (606, 348, 112, 26);
        tremRate.setBounds (606, 402, 112, 26);
        lowCut.setBounds (606, 450, 112, 26);
    }

private:
    static void chip (juce::Graphics& g, juce::Rectangle<int> area, const juce::String& label)
    {
        g.setColour (theme::panel);
        g.fillRoundedRectangle (area.toFloat(), 8.0f);
        g.setColour (theme::text);
        g.setFont (theme::font (14.0f, true));
        g.drawText (label, area, juce::Justification::centred);
    }

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
        if (! ready) message = "Nothing usable captured. Try again.";
    }

    void timerCallback() override
    {
        // stop capturing after one second of audio; only the first half second is used
        if (proc.state == SwellProcessor::recording && proc.capturedLength() > (int) proc.captureRate)
            proc.finish();
        const double tempo = proc.bpm.load() * 100.0 + proc.beatsPerBar.load();
        if (std::abs (tempo - lastTempo) > 0.001) { lastTempo = tempo; dirty = true; }
        if (lowCut.getToggleState() != lastLowCut) { lastLowCut = lowCut.getToggleState(); dirty = true; }
        if (proc.state == SwellProcessor::captured && dirty && ! juce::ModifierKeys::currentModifiers.isAnyMouseButtonDown())
            update();
        const int s = proc.state;
        if (s == SwellProcessor::armed) message = "Waiting for sound... press play in FL";
        else if (s == SwellProcessor::recording) message = "Capturing...";
        else if (s == SwellProcessor::idle) message = "Press CAPTURE, then play your sound";
        const bool busy = s == SwellProcessor::armed || s == SwellProcessor::recording;
        captureButton.setButtonText (busy ? "STOP" : "CAPTURE");
        playButton.setEnabled (ready);
        previewButton.setEnabled (ready);
        dragButton.setEnabled (ready);
        repaint();
    }

    SwellProcessor& proc;
    juce::TextButton captureButton { "CAPTURE" }, playButton { "PLAY DRY" }, previewButton { "PREVIEW SWELL" };
    DragButton dragButton;
    juce::ToggleButton lowCut;
    juce::ComboBox length, tremRate;
    juce::Slider knobs[5];
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> knobAttach[5];
    std::unique_ptr<juce::AudioProcessorValueTreeState::ComboBoxAttachment> lengthAttach, rateAttach;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lowCutAttach;
    juce::String message;
    bool ready = false, dirty = true, lastLowCut = false;
    double lastTempo = 0.0;
};

class SwellEditor : public juce::AudioProcessorEditor
{
public:
    explicit SwellEditor (SwellProcessor& p) : AudioProcessorEditor (p), panel (p)
    {
        setLookAndFeel (&look);
        addAndMakeVisible (panel);
        panel.sendLookAndFeelChange();
        setResizable (true, true);
        setResizeLimits (570, 375, 1520, 1000);
        getConstrainer()->setFixedAspectRatio (1.52);
        setSize (760, 500);
    }
    ~SwellEditor() override { setLookAndFeel (nullptr); }
    void resized() override
    {
        panel.setTransform (juce::AffineTransform::scale ((float) getWidth() / 760.0f));
    }

private:
    SwellLook look;
    SwellPanel panel;
};

juce::AudioProcessorEditor* SwellProcessor::createEditor() { return new SwellEditor (*this); }
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new SwellProcessor(); }
