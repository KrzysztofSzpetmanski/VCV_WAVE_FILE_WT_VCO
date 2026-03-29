#include "reverb_stage.hpp"

#include <cmath>

namespace reverb_stage {

float reverbTimeSecondsFromKnob(float knob) {
	const float t = clamp(knob, 0.f, 1.f);
	const float shaped = std::pow(t, 1.15f);
	const float minSec = 0.12f;
	const float maxSec = 10.0f;
	return minSec * std::pow(maxSec / minSec, shaped);
}

float reverbTimeKnobFromSeconds(float seconds) {
	const float minSec = 0.12f;
	const float maxSec = 10.0f;
	const float s = clamp(seconds, minSec, maxSec);
	const float shaped = std::log(s / minSec) / std::log(maxSec / minSec);
	return std::pow(clamp(shaped, 0.f, 1.f), 1.f / 1.15f);
}

ReverbStage::ReverbStage() {
	reset(48000.f);
}

void ReverbStage::updateReverbWetHighpass(float sampleRate) {
	constexpr float cutoffHz = 110.f;
	float sr = std::max(sampleRate, 1000.f);
	reverbWetHpCoeff = std::exp(-2.f * 3.14159265359f * cutoffHz / sr);
}

void ReverbStage::resetReverbWetHighpass() {
	reverbWetHpInL = 0.f;
	reverbWetHpInR = 0.f;
	reverbWetHpOutL = 0.f;
	reverbWetHpOutR = 0.f;
}

void ReverbStage::processReverbWetHighpass(float& wetL, float& wetR) {
	float yL = wetL - reverbWetHpInL + reverbWetHpCoeff * reverbWetHpOutL;
	float yR = wetR - reverbWetHpInR + reverbWetHpCoeff * reverbWetHpOutR;

	reverbWetHpInL = wetL;
	reverbWetHpInR = wetR;
	reverbWetHpOutL = std::isfinite(yL) ? yL : 0.f;
	reverbWetHpOutR = std::isfinite(yR) ? yR : 0.f;
	wetL = reverbWetHpOutL;
	wetR = reverbWetHpOutR;
}

void ReverbStage::reset(float sampleRate) {
	float sr = std::max(sampleRate, 1000.f);
	reverb.Init(sr);
	updateReverbWetHighpass(sr);
	resetReverbWetHighpass();
}

void ReverbStage::process(float& inOutL,
                          float& inOutR,
                          float mix,
                          float timeKnob,
                          float feedbackKnob,
                          float sampleRate) {
	float rvbMix = clamp(mix, 0.f, 1.f);
	if (rvbMix <= 1e-4f) {
		return;
	}

	float tRaw = clamp(timeKnob, 0.f, 1.f);
	float fbRaw = clamp(feedbackKnob, 0.f, 1.f);
	float rt60Sec = reverbTimeSecondsFromKnob(tRaw);
	float timeNorm = (rt60Sec - 0.12f) / (10.0f - 0.12f);
	timeNorm = clamp(timeNorm, 0.f, 1.f);

	const float rackToVerb = 0.20f;
	const float verbToRack = 1.0f / rackToVerb;

	float fbFromTime = 0.60f + 0.38f * std::pow(timeNorm, 0.90f);
	float feedback = clamp(fbFromTime + (fbRaw - 0.5f) * 0.24f, 0.45f, 0.992f);

	float damping = 0.55f * timeNorm + 0.45f * fbRaw;
	float lpHz = 16000.f - 13000.f * std::pow(damping, 0.85f);
	lpHz = clamp(lpHz, 1200.f, 18000.f);

	reverb.SetFeedback(feedback);
	reverb.SetLpFreq(lpHz);

	float dryL = clamp(inOutL * rackToVerb, -1.2f, 1.2f);
	float dryR = clamp(inOutR * rackToVerb, -1.2f, 1.2f);
	float wetL = 0.f;
	float wetR = 0.f;
	reverb.Process(dryL, dryR, &wetL, &wetR);
	processReverbWetHighpass(wetL, wetR);

	float wetGain = 1.05f + 0.55f * std::pow(fbRaw, 0.55f);
	wetL *= wetGain;
	wetR *= wetGain;

	float dryMix = std::cos(rvbMix * (0.5f * 3.14159265359f));
	float wetMix = std::sin(rvbMix * (0.5f * 3.14159265359f));
	float mixedL = (dryL * dryMix + wetL * wetMix) * verbToRack;
	float mixedR = (dryR * dryMix + wetR * wetMix) * verbToRack;

	if (!std::isfinite(mixedL) || !std::isfinite(mixedR) ||
	    std::abs(mixedL) > 40.f || std::abs(mixedR) > 40.f) {
		reset(sampleRate > 1000.f ? sampleRate : 48000.f);
		inOutL = dryL * verbToRack;
		inOutR = dryR * verbToRack;
	}
	else {
		inOutL = mixedL;
		inOutR = mixedR;
	}
}

} // namespace reverb_stage
