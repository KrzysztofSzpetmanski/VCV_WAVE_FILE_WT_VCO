#pragma once

#include "plugin.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <random>
#include <thread>
#include <vector>

class WavetableEngine {
public:
	static constexpr int kMaxWavetableSize = 4096;
	static constexpr int kGeneratedWavetableSize = 2048;
	static constexpr int kMipLevels = 5;

	WavetableEngine();
	~WavetableEngine();

	void init(int wtSize, int dens, int smoth, float scanNorm);
	void forceRebuild(int wtSize, int dens, int smoth, float scanNorm);
	void setSource(const std::shared_ptr<const std::vector<float>>& sourcePtr);
	void setTargets(int wtSize, int dens, int smoth, float scanNorm);
	void updateControl();
	void advanceBlend(float sampleTime, float transitionTimeSec);

	float readSample(float ph, float freq, float sampleRate) const;
	void copyDisplayData(std::array<float, kMaxWavetableSize>& outData, int& outSize, float& outScan) const;
	int getPublishedWtSize() const;
	float getPublishedScanNorm() const;

private:
	struct WavetableState {
		std::array<float, kGeneratedWavetableSize> wave {};
		std::array<std::array<float, kGeneratedWavetableSize>, kMipLevels> mip {};
		std::array<int, kMipLevels> mipSize {};
		int wtSize = 1024;
		float scanNorm = 0.f;
	};

	float sanitizeWaveSample(float v) const;
	float readWavetableLevelSample(const std::array<std::array<float, kGeneratedWavetableSize>, kMipLevels>& mip,
	                              const std::array<int, kMipLevels>& mipSizes,
	                              int level,
	                              float ph) const;
	void selectMipLevels(float freq, float sampleRate, int& level0, int& level1, float& blend) const;
	void rebuildMipmapsFromTable(const std::array<float, kGeneratedWavetableSize>& source,
	                             std::array<std::array<float, kGeneratedWavetableSize>, kMipLevels>& mipOut,
	                             std::array<int, kMipLevels>& mipSizesOut);
	void buildTableState(WavetableState& outState,
	                    int wtSizeParam,
	                    int dens,
	                    int smoth,
	                    float scanNorm,
	                    const std::shared_ptr<const std::vector<float>>& sourcePtr);
	int acquireBuildSlot() const;
	void submitBuildRequest(int wtSize, int dens, int smoth, float scanNorm);
	void startWorkerThread();
	void stopWorkerThread();
	void publishUiDisplayWave();

	mutable std::mt19937 rng {0x4e565f43u};

	std::array<WavetableState, 3> wavetableStates {};
	std::atomic<int> activeStateIndex {0};
	std::atomic<int> prevStateIndex {0};
	std::atomic<int> readyStateIndex {-1};
	int requestedWtSize = -1;
	int requestedDense = -1;
	int requestedSmoth = -1;
	float requestedScanNorm = -1.f;
	std::atomic<int> buildReqWtSize {1024};
	std::atomic<int> buildReqDense {100};
	std::atomic<int> buildReqSmoth {0};
	std::atomic<float> buildReqScanNorm {0.f};
	std::atomic<uint64_t> buildReqRevision {0};
	std::atomic<bool> workerRunning {false};
	std::thread workerThread;

	std::shared_ptr<const std::vector<float>> sourceMonoActive;
	std::array<std::array<float, kGeneratedWavetableSize>, 2> uiDisplayWave {};
	std::atomic<int> uiDisplayWaveIndex {0};
	std::atomic<float> uiScanNorm {0.f};
	std::atomic<int> uiWtSize {1024};
	std::atomic<float> tableBlend {1.f};
};
