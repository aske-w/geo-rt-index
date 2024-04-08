//
// Created by aske on 4/8/24.
//

#ifndef GEO_RT_INDEX_EXCEPTION_HPP
#define GEO_RT_INDEX_EXCEPTION_HPP

#include <stdexcept>

namespace geo_rt_index
{
namespace helpers
{

class Exception : public std::runtime_error
{
public:
	explicit Exception(const std::string& msg) : std::runtime_error(msg)
	{

	}
};

class ArgumentException : public Exception
{
private:
public:
	explicit ArgumentException(const std::string& argument, const std::string& msg) : Exception(std::string(argument).append(": ").append(msg))
	{

	}
};

} // helpers
} // geo_rt_index

#endif // GEO_RT_INDEX_EXCEPTION_HPP
