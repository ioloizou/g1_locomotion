cmake_minimum_required(VERSION 2.8.3)

project(g1_description)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

file(GLOB urdf "*.urdf")
file(GLOB xml "*.xml")

foreach(dir meshes)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)

foreach(filename ${urdf} ${xml})
	install(FILES ${filename}
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
endforeach(dir)
