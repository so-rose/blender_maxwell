#!/bin/bash

blender --python run.py
if [ $? -eq 42 ]; then
	echo
	echo
	echo
	echo
	echo
	blender --python run.py
fi
