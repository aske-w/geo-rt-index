//
// Created by aske on 4/14/24.
//

#ifndef GEO_RT_INDEX_DATA_LOADER_HPP
#define GEO_RT_INDEX_DATA_LOADER_HPP

#include <string>
#include <vector>
#include "types.hpp"

class DataLoader
{
public:
	static std::vector<geo_rt_index::Point> Load(const std::string& path);
};

#endif // GEO_RT_INDEX_DATA_LOADER_HPP
