#!/bin/bash

blender --python run.py
if [ $? -eq 42 ]; then
	blender --python run.py
fi
