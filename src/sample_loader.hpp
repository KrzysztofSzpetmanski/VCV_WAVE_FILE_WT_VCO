#pragma once

#include <string>
#include <vector>

namespace sample_loader {

bool loadWavMonoLimited5s(const std::string& path,
                          std::vector<float>& outMono,
                          float& outSampleRate,
                          std::string& outError);

} // namespace sample_loader
