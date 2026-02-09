### Building URL Dynamically
### Variable Rules and URL Building 

from flask import Flask, redirect, url_for

app1 = Flask(__name__)

@app1.route('/')
def hello():
    return 'Hello Guys' 

@app1.route('/passed/<int:score>')
def passed(score):
    return f'This guy has passed the test with {score} marks.'

@app1.route('/failed/<int:score>')
def failed(score):
    return f'This guy has Failed the test with {score} marks.'

@app1.route('/result/<int:score>')
def result(score):
    result= 'passed' if score > 40 else 'failed'
    return redirect(url_for(result, score=score))

if __name__ == '__main__':
    app1.run(debug=True)