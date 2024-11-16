from flask import Flask, render_template, request, redirect, session, jsonify, url_for

app = Flask(__name__)

@app.route('/')
def index():
    """
    Redirect to /home if the user is logged in.
    Otherwise, redirect to /start.
    """
    if 'profile' in session:
        # User is logged in, redirect to /home
        return redirect('/home')
    else:
        # User is not logged in, redirect to /start
        return redirect('/start')

@app.route('/start')
def start():
    """
    Render the starting page for users who are not logged in.
    """
    return render_template('start.html')

@app.route('/home')
def home():
    """
    Render the home page for logged-in users.
    """
    username = session['profile'].get('name', 'User')
    return render_template('home.html', username=username)


if __name__ == '__main__':
    app.run(debug=True)