//
// Created by aske on 4/12/24.
//

#include <bit>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <filesystem>

using std::cout, std::to_string;

static constexpr const uint8_t line_size = 17;
static constexpr const uint32_t filesize = 1 << 26;
static constexpr const uint8_t hex_length = sizeof(float) * 2;
static constexpr const uint8_t byte_length = sizeof(float);

static inline unsigned char unhex(char h){
	return ((h & 0x0f) + (h >> 6) * 9);
}

static inline unsigned hex2bin(const char* hex, void* bin){
	unsigned i, j;
	for(i = 0, j = 0; (i < byte_length) && (j+1 < hex_length); ++i, j+=2){
		unsigned char hi = unhex(hex[j+0]);
		unsigned char lo = unhex(hex[j+1]);
		((unsigned char*)bin)[i] = (hi << 4) | lo;
	}
	return i;
}

using Point = std::pair<const float, const float>;
using BBox = std::pair<const Point, const Point>;

static inline bool contains(const Point p, const BBox bounds)
{
	const auto x = p.first;
	const auto y = p.second;
	const auto minx = bounds.first.first;
	const auto miny = bounds.first.second;
	const auto maxx = bounds.second.first;
	const auto maxy = bounds.second.second;
	return minx < x && x < maxx
		&& miny < y && y < maxy;
}

int main(int argc, char** argv)
{
	uint8_t arg_index = 1;
	bool debug = false;
	bool used_for_input = false;
	auto first_arg =  argv[arg_index][0];
	switch (first_arg)
	{
	case 'd':
		debug = true;
		arg_index++;
		cout << "Debug enabled" << '\n';
	break;
	case 'i':
		used_for_input = true;
	break;
	default:
	break;
	}

	const auto minx = atof(argv[arg_index++]);
	const auto miny = atof(argv[arg_index++]);
	const auto maxx = atof(argv[arg_index++]);
	const auto maxy = atof(argv[arg_index++]);
	if(!used_for_input)
	{
		printf("Query: (%.10f,%.10f) < p < (%.10f, %.10f)\n", minx, miny, maxx, maxy);
	}
	const BBox bbox{{minx, miny}, {maxx, maxy}};
	for(uint8_t j = arg_index++; j < argc; j++)
	{
		auto path = argv[j];
		std::ifstream fs;
		char line[line_size];
		fs.open(path);
		float x;
		float y;
		unsigned temp1;
		unsigned temp2;
		uint64_t count {0};
		// #pragma omp parallel for reduction(+:count)
		for (uint32_t i = 0; i < filesize; i++)
		{
			fs.read(line, line_size);
			hex2bin(line, &temp1);
			x = std::bit_cast<float>(__bswap_constant_32(temp1));
			hex2bin(line + hex_length, &temp2);
			y = std::bit_cast<float>(__bswap_constant_32(temp2));
			if (contains({x, y}, bbox))
			{
				count++;
			}
			else
			{
				if(!debug)
				{
					continue;
				}
				printf("(%.10f, %.10f)\n", x, y);
//				if(x == 0.0f)
//					printf("(%f, %f)", x, y);

			}
		}

		//	cout << to_string(x) << " " << to_string(y) << '\n';
		const std::filesystem::path p{path};
		cout << p.filename() << ": " << to_string(count) << '\n';
	}
	return 0;
}