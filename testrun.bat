@echo off
echo ==============================================
echo [1/2] Kich hoat moi truong conda: deca5060
echo ==============================================
call conda activate deca5060

echo.
echo ==============================================
echo [2/2] Chay DECA (Su dung CUDA cho RTX 5060 Ti)
echo ==============================================
cd /d D:\deca

:: Dung --device cpu de tranh loi SM_120 cua card 50-series chua duoc PyTorch ho tro chinh thuc
:: Su dung --detector mtcnn de nhanh hon va tranh loi bien dich (cl.exe)
python demos/demo_reconstruct.py -i TestSamples/examples --device cuda --detector mtcnn --saveObj True --saveVis False --saveDepth False --saveImages False

echo.
echo Hoan Tat! Hay kiem tra cac file .obj trong TestSamples/examples/results.
pause
