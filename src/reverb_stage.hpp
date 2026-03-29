#pragma once

#include "plugin.hpp"
#include "reverbsc.h"

namespace reverb_stage {

float reverbTimeSecondsFromKnob(float knob);
float reverbTimeKnobFromSeconds(float seconds);

class ReverbStage {
public:
	ReverbStage();

	void reset(float sampleRate);
	void process(float& inOutL,
	            float& inOutR,
	            float mix,
	            float timeKnob,
	            float feedbackKnob,
	            float sampleRate);

private:
	void updateReverbWetHighpass(float sampleRate);
	void resetReverbWetHighpass();
	void processReverbWetHighpass(float& wetL, float& wetR);

	daisysp::ReverbSc reverb;
	float reverbWetHpCoeff = 0.f;
	float reverbWetHpInL = 0.f;
	float reverbWetHpInR = 0.f;
	float reverbWetHpOutL = 0.f;
	float reverbWetHpOutR = 0.f;
};

} // namespace reverb_stage
