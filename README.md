# Face-Verification

Example to run the docker container :

```powershell
docker run `
-p 8888:8888 `
--rm `
--name test `
-v C:\Users\antoi\Downloads\YouTubeFaces\:/dataset:ro `
-v .\container_output\:/output `
-v .\src\:/workspace/src `
-v C:\Users\antoi\Downloads\facenet-models\20180402-114759:/models:ro `
-it test_technique `
bash
```
``` bash
docker compose run -p 8888:8888 test_technique
``` 

``` bash
cd /scripts
python face_detection.py
```

``` bash
cd /src
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser
```