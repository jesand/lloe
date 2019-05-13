# Main library
add_library(ogt
	lib/core/basis.cpp
	lib/core/collection.cpp
	lib/core/core.cpp
	lib/core/oracle.cpp
	lib/core/splitter.cpp
	lib/core/traversal.cpp
	lib/embed/cmdet.cpp
	lib/embed/dist.cpp
	lib/embed/dlib_opt.cpp
	lib/embed/embed.cpp
	lib/embed/embed_cmp_with_ck.cpp
	lib/embed/embed_cmp_with_gnmds.cpp
	lib/embed/embed_cmp_with_pgd.cpp
	lib/embed/embed_cmp_with_soe.cpp
	lib/embed/embed_cmp_with_ste.cpp
	lib/embed/embed_cmp_with_tau.cpp
	lib/io/csv.cpp
	lib/io/io.cpp
	lib/io/svmlight.cpp
	lib/linalg/linalg.cpp
	lib/util/random.cpp
	)


# Tool: embed_cmp – Embed using comparisons
add_executable(embed_cmp
	tools/embed_cmp/embed_cmp.cpp)
target_link_libraries(embed_cmp
	ogt
	boost_program_options)

# Tool: embed_from_ref – Embed from an embedded subset
add_executable(embed_from_ref
	tools/embed_from_ref/embed_from_ref.cpp)
target_link_libraries(embed_from_ref
	ogt
	boost_program_options
	pthread)

# Tool: embed_ref – Select and embed a subset
add_executable(embed_ref
	tools/embed_ref/embed_ref.cpp)
target_link_libraries(embed_ref
	ogt
	boost_program_options
	pthread)
