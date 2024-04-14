//
// Created by aske on 4/14/24.
//

#include "helpers/data_loader.hpp"
#include "gdal_priv.h"
#include "ogr_api.h"
#include "ogrsf_frmts.h"
#include "ogr_recordbatch.h"
#include <cassert>

using std::vector;
using geo_rt_index::Point;

//! from https://gdal.org/tutorials/vector_api_tut.html#reading-from-ogr-using-the-arrow-c-stream-data-interface
vector<Point> DataLoader::Load(const std::string& path)
{
	GDALAllRegister();
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
	vector<Point> result;
	while( true )
	{
		struct ArrowArray array;
		if( stream.get_next(&stream, &array) != 0 ||
		    array.release == nullptr )
		{
			break;
		}

		assert(array.n_children == 1);

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

		result.reserve(array.length);
		for( long long i = 0; i < array.length; i++ )
		{
			if( IsSet(wkb_field_not_null, i) )
			{
				const void* wkb = wkb_field + wkb_offset[i];
				const int32_t length = wkb_offset[i+1] - wkb_offset[i];
				OGRGeometry* geom = nullptr;
				OGRGeometryFactory::createFromWkb(wkb, nullptr, &geom, length);
				assert(geom != nullptr);
				assert(geom->getGeometryType() == OGRwkbGeometryType::wkbPoint);
				const auto point = reinterpret_cast<OGRPoint*>(geom);
				const auto x = static_cast<float>(point->getX());
				const auto y = static_cast<float>(point->getY());
				result.emplace_back(x, y);
			}
			else
			{
				printf("wkb_field[%lld] = null\n", i);
			}
		}

		// Release memory taken by the batch
		array.release(&array);
	}

	// Release stream and dataset
	stream.release(&stream);
	delete poDS;
	printf("size:%zu", result.size());
	return result;
}
