@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo.
echo Training the model...
python train_model.py
echo.
echo Starting the Flask application...
python app.py
pause
