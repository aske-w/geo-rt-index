file(GLOB FILES "*.cu")
foreach (PATH ${FILES})
	get_filename_component(FILENAME ${PATH} NAME_WE)
	add_executable("plg_${FILENAME}" ${PATH})
endforeach ()