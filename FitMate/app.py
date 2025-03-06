from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

def run_exercise(script_name):
    subprocess.Popen(["python", f"scripts/{script_name}.py"])

@app.route('/pushup')
def pushup():
    run_exercise("pushup")
    return render_template('exercise.html', exercise="Pushup")

@app.route('/squat')
def squat():
    run_exercise("squat")
    return render_template('exercise.html', exercise="Squat")

@app.route('/plank')
def plank():
    run_exercise("plank")
    return render_template('exercise.html', exercise="Plank")

@app.route('/shoulder_press')
def shoulder_press():
    run_exercise("shoulder_press")
    return render_template('exercise.html', exercise="Shoulder Press")

if __name__ == '__main__':
    app.run(debug=True)
