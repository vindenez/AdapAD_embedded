#ifndef YAML_HANDLER_HPP
#define YAML_HANDLER_HPP

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

class YAMLHandler {
public:
    static std::map<std::string, std::string> parse(const std::string& filename) {
        std::map<std::string, std::string> config;
        std::ifstream file(filename);
        std::string line;
        std::vector<std::string> path;
        int current_indent = 0;
        
        // Debug output
        std::cout << "Parsing YAML file: " << filename << std::endl;
        
        while (std::getline(file, line)) {
            // Skip empty lines and comments
            if (line.empty() || line[0] == '#') continue;
            
            // Count leading spaces for indentation
            size_t indent = line.find_first_not_of(" ");
            if (indent == std::string::npos) continue;
            
            // Remove leading/trailing whitespace
            line = trim(line);
            
            // Handle indentation changes
            if (indent < current_indent) {
                int levels = (current_indent - indent) / 2;
                while (levels-- > 0 && !path.empty()) {
                    path.pop_back();
                }
            }
            current_indent = indent;
            
            // Parse key-value pairs
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string key = trim(line.substr(0, pos));
                std::string value = trim(line.substr(pos + 1));
                
                // If value is empty, this is a new section
                if (value.empty()) {
                    path.push_back(key);
                    
                    // Add an entry for the section itself
                    std::string full_key;
                    for (const auto& p : path) {
                        if (!full_key.empty()) full_key += ".";
                        full_key += p;
                    }
                    config[full_key] = "";
                } else {
                    // Build full key path
                    std::string full_key;
                    for (const auto& p : path) {
                        if (!full_key.empty()) full_key += ".";
                        full_key += p;
                    }
                    if (!full_key.empty()) full_key += ".";
                    full_key += key;
                    
                    // Remove quotes if present
                    if (!value.empty() && value[0] == '"') {
                        value = value.substr(1, value.length() - 2);
                    }
                    
                    config[full_key] = value;
                    
                }
            }
        }
        
        return config;
    }

private:
    static std::string trim(const std::string& str) {
        size_t first = str.find_first_not_of(" \t");
        if (first == std::string::npos) return "";
        size_t last = str.find_last_not_of(" \t");
        return str.substr(first, last - first + 1);
    }
};

#endif // YAML_HANDLER_HPP 