### Intergrate HTML with Flask
### HTTP verb GET and POST

### Jinja2 Template 
'''
{%...%} for statements like conditions and for loops 
{{  }} expressions to print output
{#   #} this is for comments 
'''

from flask import Flask, redirect, url_for, render_template, request

app2 = Flask(__name__)

@app2.route('/')
def hello():
    return render_template('index.html')

@app2.route('/passed/<int:score>')
def passed(score):
    res = ''
    if score>=50:
        res='PASS'
    else:
        res='FAIL'
    exp={'score': score, 'res': res}    
    return render_template('result.html', result=exp)

@app2.route('/failed/<int:score>')
def failed(score):
    return f'This guy has Failed the test with {score} marks.'

@app2.route('/result/<int:score>')
def result(score):
    result= 'passed' if score > 40 else 'failed'
    return redirect(url_for(result, score=score))

### Result Checker HTML Page
@app2.route('/submit', methods=['POST', 'GET'])
def submit():
    total_score= 0
    if request.method=='POST':
        science=float(request.form['science'])
        maths=float(request.form['maths'])
        c=float(request.form['c'])
        data_science=float(request.form['datascience'])
        total_score=(science+maths+c+data_science)/4
    
    return redirect(url_for('passed', score=total_score))
    

if __name__ == '__main__':
    app2.run(debug=True)