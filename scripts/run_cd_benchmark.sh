#!/bin/bash
python tt_raa_runner.py --config configs \
                        --data-root "/Data/dataset" \
                        --datasets fgvc/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101 \
                        --backbone ViT-B/16