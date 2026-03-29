#include "wavetable_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

WavetableEngine::WavetableEngine() {
}

WavetableEngine::~WavetableEngine() {
	stopWorkerThread();
}

void WavetableEngine::init(int wtSize, int dens, int smoth, float scanNorm) {
	buildTableState(
		wavetableStates[0],
		wtSize,
		dens,
		smoth,
		scanNorm,
		std::shared_ptr<const std::vector<float>> {}
	);
	wavetableStates[1] = wavetableStates[0];
	wavetableStates[2] = wavetableStates[0];
	activeStateIndex.store(0, std::memory_order_relaxed);
	prevStateIndex.store(0, std::memory_order_relaxed);
	readyStateIndex.store(-1, std::memory_order_relaxed);
	requestedWtSize = -1;
	requestedDense = -1;
	requestedSmoth = -1;
	requestedScanNorm = -1.f;
	tableBlend.store(1.f, std::memory_order_relaxed);
	publishUiDisplayWave();
	setTargets(wtSize, dens, smoth, scanNorm);
	startWorkerThread();
}

void WavetableEngine::forceRebuild(int wtSize, int dens, int smoth, float scanNorm) {
	int active = activeStateIndex.load(std::memory_order_acquire);
	prevStateIndex.store(active, std::memory_order_release);
	readyStateIndex.store(-1, std::memory_order_release);
	tableBlend.store(1.f, std::memory_order_relaxed);
	requestedWtSize = -1;
	requestedDense = -1;
	requestedSmoth = -1;
	requestedScanNorm = -1.f;
	setTargets(wtSize, dens, smoth, scanNorm);
	publishUiDisplayWave();
}

void WavetableEngine::setSource(const std::shared_ptr<const std::vector<float>>& sourcePtr) {
	std::atomic_store_explicit(&sourceMonoActive, sourcePtr, std::memory_order_release);
	submitBuildRequest(
		(requestedWtSize >= 0) ? requestedWtSize : 1024,
		(requestedDense >= 0) ? requestedDense : 100,
		(requestedSmoth >= 0) ? requestedSmoth : 0,
		(requestedScanNorm >= 0.f) ? requestedScanNorm : 0.f
	);
}

void WavetableEngine::setTargets(int wtSize, int dens, int smoth, float scanNorm) {
	int wt = clamp(wtSize, 256, kGeneratedWavetableSize);
	int de = clamp(dens, 0, 100);
	int sm = clamp(smoth, 0, 100);
	float sc = clamp(scanNorm, 0.f, 1.f);
	bool changed = (wt != requestedWtSize) ||
	               (de != requestedDense) ||
	               (sm != requestedSmoth) ||
	               (std::abs(sc - requestedScanNorm) > 1e-4f);
	if (!changed) {
		return;
	}
	requestedWtSize = wt;
	requestedDense = de;
	requestedSmoth = sm;
	requestedScanNorm = sc;
	submitBuildRequest(wt, de, sm, sc);
}

void WavetableEngine::updateControl() {
	int ready = readyStateIndex.exchange(-1, std::memory_order_acq_rel);
	if (ready >= 0) {
		int active = activeStateIndex.load(std::memory_order_acquire);
		if (ready != active) {
			prevStateIndex.store(active, std::memory_order_release);
			activeStateIndex.store(ready, std::memory_order_release);
			tableBlend.store(0.f, std::memory_order_relaxed);
		}
		publishUiDisplayWave();
	}

	float blend = tableBlend.load(std::memory_order_relaxed);
	if (blend >= 0.999f) {
		int active = activeStateIndex.load(std::memory_order_acquire);
		prevStateIndex.store(active, std::memory_order_release);
	}
	if (blend < 0.999f) {
		publishUiDisplayWave();
	}
}

void WavetableEngine::advanceBlend(float sampleTime, float transitionTimeSec) {
	float blend = tableBlend.load(std::memory_order_relaxed);
	blend = clamp(blend + sampleTime / std::max(transitionTimeSec, 1e-4f), 0.f, 1.f);
	tableBlend.store(blend, std::memory_order_relaxed);
}

float WavetableEngine::sanitizeWaveSample(float v) const {
	if (!std::isfinite(v)) {
		return 0.f;
	}
	return clamp(v, -1.f, 1.f);
}

float WavetableEngine::readWavetableLevelSample(const std::array<std::array<float, kGeneratedWavetableSize>, kMipLevels>& mip,
                                                const std::array<int, kMipLevels>& mipSizes,
                                                int level,
                                                float ph) const {
	auto wrapIndex = [](int i, int size) {
		int r = i % size;
		return (r < 0) ? (r + size) : r;
	};
	auto hermite4 = [](float xm1, float x0, float x1, float x2, float t) {
		float c0 = x0;
		float c1 = 0.5f * (x1 - xm1);
		float c2 = xm1 - 2.5f * x0 + 2.f * x1 - 0.5f * x2;
		float c3 = 0.5f * (x2 - xm1) + 1.5f * (x0 - x1);
		return ((c3 * t + c2) * t + c1) * t + c0;
	};

	int l = clamp(level, 0, kMipLevels - 1);
	int sizeLocal = mipSizes[l];
	if (sizeLocal < 2) {
		return 0.f;
	}

	float pos = ph * static_cast<float>(sizeLocal - 1);
	int i0 = static_cast<int>(std::floor(pos));
	int i1 = wrapIndex(i0 + 1, sizeLocal);
	float frac = pos - static_cast<float>(i0);

	i0 = wrapIndex(i0, sizeLocal);
	float s0 = mip[l][i0];
	float s1 = mip[l][i1];
	float linear = s0 + (s1 - s0) * frac;
	if (sizeLocal < 4) {
		return sanitizeWaveSample(linear);
	}

	int im1 = wrapIndex(i0 - 1, sizeLocal);
	int i2 = wrapIndex(i0 + 2, sizeLocal);
	float cubic = hermite4(mip[l][im1], s0, s1, mip[l][i2], frac);
	return sanitizeWaveSample(cubic);
}

void WavetableEngine::selectMipLevels(float freq, float sampleRate, int& level0, int& level1, float& blend) const {
	float sr = std::max(sampleRate, 1000.f);
	float f = std::max(std::abs(freq), 1.f);
	float desiredSize = clamp(sr / f, 128.f, static_cast<float>(kGeneratedWavetableSize));
	float rawLevel = std::log2(static_cast<float>(kGeneratedWavetableSize) / desiredSize);
	float levelF = clamp(rawLevel + 0.20f, 0.f, static_cast<float>(kMipLevels - 1));
	level0 = static_cast<int>(std::floor(levelF));
	level1 = std::min(level0 + 1, kMipLevels - 1);
	blend = levelF - static_cast<float>(level0);
}

float WavetableEngine::readSample(float ph, float freq, float sampleRate) const {
	int level0 = 0;
	int level1 = 0;
	float levelBlend = 0.f;
	selectMipLevels(freq, sampleRate, level0, level1, levelBlend);

	int activeIdx = activeStateIndex.load(std::memory_order_acquire);
	int prevIdx = prevStateIndex.load(std::memory_order_acquire);
	const WavetableState& active = wavetableStates[activeIdx];
	const WavetableState& prevState = wavetableStates[prevIdx];

	float curr0 = readWavetableLevelSample(active.mip, active.mipSize, level0, ph);
	float curr1 = readWavetableLevelSample(active.mip, active.mipSize, level1, ph);
	float curr = sanitizeWaveSample(curr0 + (curr1 - curr0) * levelBlend);

	float prev0 = readWavetableLevelSample(prevState.mip, prevState.mipSize, level0, ph);
	float prev1 = readWavetableLevelSample(prevState.mip, prevState.mipSize, level1, ph);
	float prev = sanitizeWaveSample(prev0 + (prev1 - prev0) * levelBlend);

	float blend = clamp(tableBlend.load(std::memory_order_relaxed), 0.f, 1.f);
	return sanitizeWaveSample(prev + (curr - prev) * blend);
}

void WavetableEngine::rebuildMipmapsFromTable(const std::array<float, kGeneratedWavetableSize>& source,
                                              std::array<std::array<float, kGeneratedWavetableSize>, kMipLevels>& mipOut,
                                              std::array<int, kMipLevels>& mipSizesOut) {
	const int baseSize = kGeneratedWavetableSize;
	mipSizesOut[0] = baseSize;
	for (int i = 0; i < baseSize; ++i) {
		mipOut[0][i] = sanitizeWaveSample(source[i]);
	}
	mipOut[0][0] = 0.f;
	mipOut[0][baseSize - 1] = 0.f;

	for (int level = 1; level < kMipLevels; ++level) {
		int prevSize = mipSizesOut[level - 1];
		int size = std::max(128, prevSize / 2);
		mipSizesOut[level] = size;

		for (int i = 0; i < size; ++i) {
			int i0 = std::min(i * 2, prevSize - 1);
			int i1 = std::min(i0 + 1, prevSize - 1);
			float a = mipOut[level - 1][i0];
			float b = mipOut[level - 1][i1];
			mipOut[level][i] = sanitizeWaveSample(0.5f * (a + b));
		}

		for (int i = size; i < kGeneratedWavetableSize; ++i) {
			mipOut[level][i] = 0.f;
		}
		mipOut[level][0] = 0.f;
		mipOut[level][size - 1] = 0.f;
	}
}

void WavetableEngine::buildTableState(WavetableState& outState,
                                      int wtSizeParam,
                                      int dens,
                                      int smoth,
                                      float scanNorm,
                                      const std::shared_ptr<const std::vector<float>>& sourcePtr) {
	float smothF = clamp(smoth * 0.01f, 0.f, 1.f);
	float densF = clamp(dens * 0.01f, 0.f, 1.f);
	int windowFrames = clamp(wtSizeParam, 256, kGeneratedWavetableSize);
	bool sourceReady = (sourcePtr && sourcePtr->size() > 2);
	std::array<float, kGeneratedWavetableSize> localWindow {};

	if (sourceReady) {
		int srcSize = static_cast<int>(sourcePtr->size());
		windowFrames = std::min(windowFrames, srcSize);
		int maxStart = std::max(0, srcSize - windowFrames);
		int start = static_cast<int>(std::lround(clamp(scanNorm, 0.f, 1.f) * static_cast<float>(maxStart)));
		start = clamp(start, 0, maxStart);
		for (int i = 0; i < windowFrames; ++i) {
			localWindow[i] = sanitizeWaveSample((*sourcePtr)[start + i]);
		}
	}
	else {
		std::uniform_real_distribution<float> dist(-1.f, 1.f);
		for (int i = 0; i < windowFrames; ++i) {
			localWindow[i] = dist(rng);
		}
	}

	int truePoints = static_cast<int>(std::lround(64.f + densF * static_cast<float>(windowFrames - 64)));
	truePoints = clamp(truePoints, 64, windowFrames);

	std::array<int, kGeneratedWavetableSize> anchorIdx {};
	int prev = 0;
	for (int p = 0; p < truePoints; ++p) {
		float t = (truePoints <= 1) ? 0.f : static_cast<float>(p) / static_cast<float>(truePoints - 1);
		int idx = static_cast<int>(std::lround(t * static_cast<float>(windowFrames - 1)));
		int remaining = truePoints - 1 - p;
		int maxAllowed = (windowFrames - 1) - remaining;
		idx = clamp(idx, prev, maxAllowed);
		anchorIdx[p] = idx;
		prev = idx + 1;
	}

	std::array<float, kGeneratedWavetableSize> simplified = localWindow;
	const float pi = 3.14159265359f;
	for (int p = 0; p < truePoints - 1; ++p) {
		int i0 = anchorIdx[p];
		int i1 = anchorIdx[p + 1];
		float y0 = localWindow[i0];
		float y1 = localWindow[i1];
		int span = std::max(1, i1 - i0);
		for (int i = i0; i <= i1; ++i) {
			float t = static_cast<float>(i - i0) / static_cast<float>(span);
			float linear = y0 + (y1 - y0) * t;
			float sinusT = 0.5f - 0.5f * std::cos(pi * t);
			float sinus = y0 + (y1 - y0) * sinusT;
			simplified[i] = sanitizeWaveSample(linear + (sinus - linear) * smothF);
		}
	}
	for (int p = 0; p < truePoints; ++p) {
		int idx = anchorIdx[p];
		simplified[idx] = localWindow[idx];
	}
	localWindow = simplified;

	const float outDen = static_cast<float>(kGeneratedWavetableSize - 1);
	const float srcDen = static_cast<float>(windowFrames - 1);
	for (int i = 0; i < kGeneratedWavetableSize; ++i) {
		float pos = (static_cast<float>(i) / outDen) * srcDen;
		int i0 = static_cast<int>(std::floor(pos));
		int i1 = std::min(i0 + 1, windowFrames - 1);
		float frac = pos - static_cast<float>(i0);
		float s0 = localWindow[i0];
		float s1 = localWindow[i1];
		outState.wave[i] = sanitizeWaveSample(s0 + (s1 - s0) * frac);
	}

	const int edgeSamples = 64;
	const int edge = std::min(edgeSamples, kGeneratedWavetableSize / 2);
	for (int i = 0; i < edge; ++i) {
		float t = static_cast<float>(i) / static_cast<float>(std::max(1, edge - 1));
		float w = 0.5f - 0.5f * std::cos(pi * t);
		outState.wave[i] = sanitizeWaveSample(outState.wave[i] * w);
		outState.wave[kGeneratedWavetableSize - 1 - i] = sanitizeWaveSample(outState.wave[kGeneratedWavetableSize - 1 - i] * w);
	}
	outState.wave[0] = 0.f;
	outState.wave[kGeneratedWavetableSize - 1] = 0.f;
	outState.wtSize = windowFrames;
	outState.scanNorm = clamp(scanNorm, 0.f, 1.f);
	rebuildMipmapsFromTable(outState.wave, outState.mip, outState.mipSize);
}

int WavetableEngine::acquireBuildSlot() const {
	int active = activeStateIndex.load(std::memory_order_acquire);
	int prev = prevStateIndex.load(std::memory_order_acquire);
	int ready = readyStateIndex.load(std::memory_order_acquire);
	for (int i = 0; i < static_cast<int>(wavetableStates.size()); ++i) {
		if (i != active && i != prev && i != ready) {
			return i;
		}
	}
	return -1;
}

void WavetableEngine::submitBuildRequest(int wtSize, int dens, int smoth, float scanNorm) {
	buildReqWtSize.store(clamp(wtSize, 256, kGeneratedWavetableSize), std::memory_order_relaxed);
	buildReqDense.store(clamp(dens, 0, 100), std::memory_order_relaxed);
	buildReqSmoth.store(clamp(smoth, 0, 100), std::memory_order_relaxed);
	buildReqScanNorm.store(clamp(scanNorm, 0.f, 1.f), std::memory_order_relaxed);
	buildReqRevision.fetch_add(1, std::memory_order_release);
}

void WavetableEngine::startWorkerThread() {
	if (workerRunning.exchange(true, std::memory_order_acq_rel)) {
		return;
	}
	workerThread = std::thread([this]() {
		uint64_t seenRevision = buildReqRevision.load(std::memory_order_acquire);
		while (workerRunning.load(std::memory_order_acquire)) {
			uint64_t revision = buildReqRevision.load(std::memory_order_acquire);
			if (revision == seenRevision) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			int slot = acquireBuildSlot();
			if (slot < 0) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			int wtSize = buildReqWtSize.load(std::memory_order_relaxed);
			int dens = buildReqDense.load(std::memory_order_relaxed);
			int smoth = buildReqSmoth.load(std::memory_order_relaxed);
			float scanNorm = buildReqScanNorm.load(std::memory_order_relaxed);
			auto sourcePtr = std::atomic_load_explicit(&sourceMonoActive, std::memory_order_acquire);
			buildTableState(wavetableStates[slot], wtSize, dens, smoth, scanNorm, sourcePtr);

			uint64_t revisionAfterBuild = buildReqRevision.load(std::memory_order_acquire);
			if (revisionAfterBuild != revision) {
				seenRevision = revision;
				continue;
			}

			readyStateIndex.store(slot, std::memory_order_release);
			seenRevision = revision;
		}
	});
}

void WavetableEngine::stopWorkerThread() {
	workerRunning.store(false, std::memory_order_release);
	if (workerThread.joinable()) {
		workerThread.join();
	}
}

void WavetableEngine::publishUiDisplayWave() {
	int next = 1 - uiDisplayWaveIndex.load(std::memory_order_relaxed);
	float blend = clamp(tableBlend.load(std::memory_order_relaxed), 0.f, 1.f);
	int activeIdx = activeStateIndex.load(std::memory_order_acquire);
	int prevIdx = prevStateIndex.load(std::memory_order_acquire);
	const WavetableState& active = wavetableStates[activeIdx];
	const WavetableState& prev = wavetableStates[prevIdx];
	for (int i = 0; i < kGeneratedWavetableSize; ++i) {
		float prevSample = prev.wave[i];
		float currSample = active.wave[i];
		uiDisplayWave[next][i] = sanitizeWaveSample(prevSample + (currSample - prevSample) * blend);
	}
	uiDisplayWaveIndex.store(next, std::memory_order_release);
	uiScanNorm.store(active.scanNorm, std::memory_order_relaxed);
	uiWtSize.store(active.wtSize, std::memory_order_relaxed);
}

void WavetableEngine::copyDisplayData(std::array<float, kMaxWavetableSize>& outData, int& outSize, float& outScan) const {
	int idx = uiDisplayWaveIndex.load(std::memory_order_acquire);
	outSize = kGeneratedWavetableSize;
	for (int i = 0; i < outSize; ++i) {
		outData[i] = uiDisplayWave[idx][i];
	}
	outScan = uiScanNorm.load(std::memory_order_relaxed);
}

int WavetableEngine::getPublishedWtSize() const {
	return uiWtSize.load(std::memory_order_relaxed);
}

float WavetableEngine::getPublishedScanNorm() const {
	return uiScanNorm.load(std::memory_order_relaxed);
}
