FILE_ID=1c5ZTuT7J08wLUoVZ2KkUs_VdZuJ86ZqA
FILE_NAME=tacotron2_statedict.pt
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${FILE_NAME}
