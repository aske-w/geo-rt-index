//
// Created by aske on 4/14/24.
//

#include "gdal_priv.h"
#include "helpers/data_loader.hpp"
#include "ogr_api.h"
#include "ogr_recordbatch.h"
#include "ogrsf_frmts.h"
#include <semaphore>
#include <thread>
#include <cassert>
#include <future>
#include <string>
#include "helpers/debug_helpers.hpp"

using geo_rt_index::types::Point;

using std::vector;

static std::counting_semaphore load_limiter{std::ptrdiff_t{std::thread::hardware_concurrency()}};

static vector<Point> Work(const std::string& path, const float modifier) noexcept
{
	load_limiter.acquire();
	GDALDataset* poDS = GDALDataset::Open(path.c_str());
	if( poDS == nullptr )
	{
		CPLError(CE_Failure, CPLE_AppDefined, "Open() failed\n");
		exit(1);
	}
	OGRLayer* poLayer = poDS->GetLayer(0);
	OGRLayerH hLayer = OGRLayer::ToHandle(poLayer);

	// Get the Arrow stream
	struct ArrowArrayStream stream;
	if( !OGR_L_GetArrowStream(hLayer, &stream, nullptr))
	{
		CPLError(CE_Failure, CPLE_AppDefined, "OGR_L_GetArrowStream() failed\n");
		delete poDS;
		exit(1);
	}

	// Get the schema
	struct ArrowSchema schema;
	if( stream.get_schema(&stream, &schema) != 0 )
	{
		CPLError(CE_Failure, CPLE_AppDefined, "get_schema() failed\n");
		stream.release(&stream);
		delete poDS;
		exit(1);
	}

	// Check that the returned schema consists of one int64 field (for FID),
	// one int32 field and one binary/wkb field
	if( schema.n_children != 1 || strcmp(schema.children[0]->format, "z") != 0 )  // binary for WKB
	{
		CPLError(CE_Failure, CPLE_AppDefined,
		         "Layer has not the expected schema required by this example.");
		schema.release(&schema);
		stream.release(&stream);
		delete poDS;
		exit(1);
	}
	schema.release(&schema);

	// Iterate over batches




	vector<Point> private_result;
	while(true)
	{
		ArrowArray array;
		if( stream.get_next(&stream, &array) != 0 ||
		    array.release == nullptr )
		{
			break;
		}

		const auto wkb_child = array.children[0];
		assert(wkb_child->n_buffers == 3);
		const uint8_t* wkb_field_not_null = static_cast<const uint8_t*>(wkb_child->buffers[0]);
		const int32_t* wkb_offset = static_cast<const int32_t*>(wkb_child->buffers[1]);
		const uint8_t* wkb_field = static_cast<const uint8_t*>(wkb_child->buffers[2]);

		// Lambda to check if a field is set for a given feature index
		const auto IsSet = [](const uint8_t* buffer_not_null, int i)
		{
			return buffer_not_null == nullptr || (buffer_not_null[i/8] >> (i%8)) != 0;
		};

		private_result.reserve(array.length);
		for( long long i = 0; i < array.length; i++ )
		{
			if( !IsSet(wkb_field_not_null, i) )
			{
				throw std::runtime_error("wkb field is null: " + std::to_string(i));
			}
			//				{
			const void* wkb = wkb_field + wkb_offset[i];
			const int32_t length = wkb_offset[i+1] - wkb_offset[i];
			OGRPoint point;
			size_t consumed_bytes;
			point.importFromWkb(reinterpret_cast<const unsigned char*>(wkb), length, wkbVariantOldOgc, consumed_bytes);
			//					OGRGeometryFactory::createFromWkb(wkb, nullptr, reinterpret_cast<OGRGeometry**>(&point), length);
			//					assert(geom != nullptr);
			//					assert(geom->getGeometryType() == OGRwkbGeometryType::wkbPoint);
			const auto x = static_cast<float>(point.getX()) * modifier;
			const auto y = static_cast<float>(point.getY()) * modifier;
			private_result.emplace_back(x, y);
			//					delete point;
			//				}
			//				else
			//				{
			//					printf("wkb_field[%lld] = null\n", i);
			//				}
		}
		// Release memory taken by the batch
		array.release(&array);
	}
	load_limiter.release();
	// Release stream and dataset
	stream.release(&stream);
	delete poDS;
	return private_result;
}

//! from https://gdal.org/tutorials/vector_api_tut.html#reading-from-ogr-using-the-arrow-c-stream-data-interface
vector<Point> DataLoader::Load(const vector<std::string>& files, const float modifier)
{
	OGRRegisterAll();
	vector<Point> result;
	vector<std::future<vector<Point>>> futures;
	for(auto path : files)
	{
		auto handle = std::async(std::launch::async, Work, path, modifier);
		futures.push_back(std::move(handle));
	}

	for(auto&& handle : futures)
	{
		auto future_result = handle.get();
		result.insert(result.end(), future_result.begin(), future_result.end());
	}
	D_PRINT("size: %zu\n", result.size());
	OGRCleanupAll();
	return result;
}
