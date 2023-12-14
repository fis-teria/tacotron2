if [ $1 = "jupyter" ]; then
	sudo docker run --gpus all -v $(pwd):/content -u jupyter -p 10000:8888 -it local/nvidia_tf2_tts1_colab:$2
elif [ $1 = "0" ]; then
	sudo docker run --gpus all -v $(pwd):/content -u 0 -p 10000:8888 -it local/nvidia_tf2_tts1_colab:$2
else
	echo "Plese write bash run_ML.sh username(jupyter or 0(root)) vesion"
fi


