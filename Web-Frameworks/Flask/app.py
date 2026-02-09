from flask import Flask

# WSGI Application
app=Flask(__name__)


@app.route('/')
def welcome():
    return 'Ente manayileke swakatham. hello pookie'

@app.route('/skibidi')
def sarvam_maya():
    return 'Nala thanutha kaattundale!'

if __name__=='__main__':
    app.run(debug=True)