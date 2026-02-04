echo "Starting landmark prediction..."
python 1_get_landmarks.py --input examples/lato_metafile_input.json --output examples/results/lato_metafile_output1.json
echo "Landmark prediction done. Starting landmark prediction with VLM..."
python 2_landmark_predictor.py --input examples/results/lato_metafile_output1.json --output examples/results/lato_metafile_output2.json --prompt_file ./prompt_template.txt --lp_model_path models/LaTo/Landmark_Predictor --target_landmark_image_path examples/results/target_landmark/
echo "Landmark prediction with VLM done. Starting LaTo inference..."
python 3_lato_inference.py --model_path models/LaTo/lato/ --lm_ae_path models/LaTo/Po_VQVAE/model.safetensors --output_dir examples/results/edited_images --seed -1 --size_level 512 --landmark_path examples/results/lato_metafile_output2.json --lora models/LaTo/lora/lato.safetensors --qwen2vl_model_path models/Qwen/Qwen2.5-VL-7B-Instruct --repeat 1
echo "All steps done."