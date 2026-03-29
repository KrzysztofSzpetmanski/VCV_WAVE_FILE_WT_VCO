#include "sample_loader.hpp"

#include "plugin.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

namespace sample_loader {

static uint16_t readU16LE(const uint8_t* p) {
	return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

static uint32_t readU32LE(const uint8_t* p) {
	return static_cast<uint32_t>(p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24));
}

bool loadWavMonoLimited5s(const std::string& path,
                          std::vector<float>& outMono,
                          float& outSampleRate,
                          std::string& outError) {
	outMono.clear();
	outSampleRate = 0.f;
	outError.clear();

	std::ifstream f(path, std::ios::binary);
	if (!f.is_open()) {
		outError = "Cannot open file";
		return false;
	}

	f.seekg(0, std::ios::end);
	std::streamoff fileSize = f.tellg();
	if (fileSize < 44) {
		outError = "File too short";
		return false;
	}
	f.seekg(0, std::ios::beg);

	std::vector<uint8_t> bytes(static_cast<size_t>(fileSize));
	if (!f.read(reinterpret_cast<char*>(bytes.data()), fileSize)) {
		outError = "Read error";
		return false;
	}

	if (std::memcmp(bytes.data(), "RIFF", 4) != 0 || std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
		outError = "Not a RIFF/WAVE file";
		return false;
	}

	bool haveFmt = false;
	bool haveData = false;
	uint16_t audioFormat = 0;
	uint16_t channels = 0;
	uint16_t bitsPerSample = 0;
	uint32_t sampleRate = 0;
	const uint8_t* dataPtr = nullptr;
	size_t dataSize = 0;

	size_t offset = 12;
	while (offset + 8 <= bytes.size()) {
		const uint8_t* chunk = bytes.data() + offset;
		uint32_t chunkSize = readU32LE(chunk + 4);
		size_t chunkDataOffset = offset + 8;
		size_t chunkNext = chunkDataOffset + static_cast<size_t>(chunkSize);
		if (chunkNext > bytes.size()) {
			break;
		}

		if (std::memcmp(chunk, "fmt ", 4) == 0 && chunkSize >= 16) {
			const uint8_t* fmt = bytes.data() + chunkDataOffset;
			audioFormat = readU16LE(fmt + 0);
			channels = readU16LE(fmt + 2);
			sampleRate = readU32LE(fmt + 4);
			bitsPerSample = readU16LE(fmt + 14);
			haveFmt = true;
		}
		else if (std::memcmp(chunk, "data", 4) == 0) {
			dataPtr = bytes.data() + chunkDataOffset;
			dataSize = static_cast<size_t>(chunkSize);
			haveData = true;
		}

		offset = chunkNext + (chunkSize & 1u);
	}

	if (!haveFmt || !haveData) {
		outError = "Missing fmt/data chunk";
		return false;
	}
	if (sampleRate < 1000 || channels < 1) {
		outError = "Unsupported sample rate/channels";
		return false;
	}

	const bool isPcmInt = (audioFormat == 1);
	const bool isFloat = (audioFormat == 3);
	if (!isPcmInt && !isFloat) {
		outError = "Unsupported WAV format (only PCM/float)";
		return false;
	}

	int bytesPerSample = bitsPerSample / 8;
	if (bytesPerSample < 1) {
		outError = "Unsupported bit depth";
		return false;
	}

	if (isPcmInt && !(bitsPerSample == 16 || bitsPerSample == 24 || bitsPerSample == 32)) {
		outError = "Unsupported PCM bit depth";
		return false;
	}
	if (isFloat && bitsPerSample != 32) {
		outError = "Unsupported float bit depth";
		return false;
	}

	size_t frameBytes = static_cast<size_t>(bytesPerSample) * static_cast<size_t>(channels);
	if (frameBytes == 0 || dataSize < frameBytes) {
		outError = "Invalid data chunk";
		return false;
	}

	size_t totalFrames = dataSize / frameBytes;
	size_t maxFrames = std::min(totalFrames, static_cast<size_t>(sampleRate) * 5u);
	if (maxFrames < 2) {
		outError = "Too few audio frames";
		return false;
	}

	outMono.reserve(maxFrames);
	for (size_t frame = 0; frame < maxFrames; ++frame) {
		const uint8_t* framePtr = dataPtr + frame * frameBytes;
		float sum = 0.f;
		for (uint16_t ch = 0; ch < channels; ++ch) {
			const uint8_t* s = framePtr + static_cast<size_t>(ch) * static_cast<size_t>(bytesPerSample);
			float v = 0.f;
			if (isPcmInt) {
				if (bitsPerSample == 16) {
					int16_t x = static_cast<int16_t>(s[0] | (s[1] << 8));
					v = static_cast<float>(x) / 32768.f;
				}
				else if (bitsPerSample == 24) {
					int32_t x = static_cast<int32_t>(s[0] | (s[1] << 8) | (s[2] << 16));
					if (x & 0x00800000) {
						x |= ~0x00FFFFFF;
					}
					v = static_cast<float>(x) / 8388608.f;
				}
				else {
					int32_t x = static_cast<int32_t>(readU32LE(s));
					v = static_cast<float>(x) / 2147483648.f;
				}
			}
			else {
				float x = 0.f;
				std::memcpy(&x, s, sizeof(float));
				v = x;
			}
			sum += v;
		}
		outMono.push_back(clamp(sum / static_cast<float>(channels), -1.f, 1.f));
	}

	outSampleRate = static_cast<float>(sampleRate);
	return true;
}

} // namespace sample_loader
