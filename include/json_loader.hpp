#ifndef JSON_LOADER_HPP
#define JSON_LOADER_HPP

#include <string>
#include <vector>
#include <unordered_map>

std::vector<std::vector<float>> load_weights(const std::string& filename, const std::string& key);
std::vector<float> load_bias(const std::string& filename, const std::string& key);
std::unordered_map<std::string, std::vector<std::vector<float>>> load_all_weights(const std::string& filename);
std::unordered_map<std::string, std::vector<float>> load_all_biases(const std::string& filename);

#endif // JSON_LOADER_HPP
