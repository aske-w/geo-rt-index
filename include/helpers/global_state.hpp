
#ifndef GEO_RT_INDEX_METADATA_HPP
#define GEO_RT_INDEX_METADATA_HPP

namespace geo_rt_index
{
namespace helpers
{
    
class GlobalState
{
private:
    bool is_warmup = false;
    
	GlobalState() {};
	inline static GlobalState& GetInstance()
	{
		static GlobalState instance{};
		return instance;
	}
public:
    GlobalState(GlobalState&) = delete;
    void operator=(const GlobalState&) = delete; 
    static inline void SetIsWarmup(bool value)
    {
        GetInstance().is_warmup = value;
    }
    static inline bool GetIsWarmup()
    { 
        return GetInstance().is_warmup;
    }
};
    
} // namespace helpers
} // namespace geo_rt_index


#endif //GEO_RT_INDEX_METADATA_HPP
