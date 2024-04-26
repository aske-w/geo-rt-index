//
// Created by aske on 4/8/24.
//

#ifndef GEO_RT_INDEX_GENERAL_HPP
#define GEO_RT_INDEX_GENERAL_HPP

#include <memory>
#include <string>
#include <stdexcept>

namespace geo_rt_index
{
namespace helpers
{

#define nameof(arg) #arg

//! from https://stackoverflow.com/a/26221725
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
	const int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
	if( size_s <= 0 )
	{
		throw Exception{"Error during formatting."};
	}
	const auto size = static_cast<size_t>( size_s );
	std::unique_ptr<char[]> buf( new char[ size ] );
	std::snprintf( buf.get(), size, format.c_str(), args ... );
	return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

} // helpers
} // geo_rt_index


#endif // GEO_RT_INDEX_GENERAL_HPP
