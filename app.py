from flask import Flask, render_template, request, flash

import summaryEx as se
import summaryAb as sa
import keyWord as kw
import translate as tr
import paraphrase as pa
import sentiment as sent

import os

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = '8f42a73054b1749f8f58848be5e6502c'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/summ_ab', methods=['POST', 'GET'])
def summ_ab():
    if request.method == 'POST':
        text = request.form['content']

        if not text:
            flash('Please enter text')
        else:

            res = sa.main(text)
            return render_template('summ_ab.html', result=res)

    return render_template('summ_ab.html')


@app.route('/summ_ex', methods=['POST', 'GET'])
def summ_ex():
    if request.method == 'POST':
        text = request.form['content']
        n_sentence = int(request.form['n_sentence'])
        maxLength = int(request.form['maxLength'])
        n_topic = int(request.form['n_topic'])

        if not text:
            flash('Please enter text')
        else:
            res, plot1, plot2 = se.main(text, n_sentence, maxLength, n_topic)
            return render_template('summ_ex.html', result=res, url_plot1=plot1, url_plot2=plot2)

    return render_template('summ_ex.html')


@app.route('/keywords', methods=['POST', 'GET'])
def keywords():
    if request.method == 'POST':
        text = request.form['content']

        if not text:
            flash('Please enter text')
        else:
            res = kw.main(text)
            return render_template('keywords.html', result=res)

    return render_template('keywords.html')


@app.route('/translate', methods=['POST', 'GET'])
def translate():
    data = [{'langue': 'French'}, {'langue': 'English'}, {'langue': 'German'}, {'langue': 'Romanian'}]

    if request.method == 'POST':
        text = request.form['content']
        langueEntry = request.form['langueEntry']
        langueExit = request.form['langueExit']

        if not text:
            flash('Please enter text')
        elif langueEntry == langueExit:
            flash('Please select different languages')
        else:
            res = tr.main(text, langueEntry, langueExit)
            return render_template('translate.html', data=data, result=res)

    return render_template('translate.html',
                           data=data)


@app.route('/paraphrase', methods=['POST', 'GET'])
def paraphrase():
    data = [{'type': 'URL'}, {'type': 'Text'}]

    if request.method == 'POST':
        text = request.form['content']
        type = request.form['type']

        if not text:
            flash('Please enter text')
        else:
            res, lang = pa.main(text, type)
            return render_template('paraphrase.html', result=res, lang=lang, data=data)

    return render_template('paraphrase.html', data=data)


@app.route('/sentiment', methods=['POST', 'GET'])
def sentiment():
    if request.method == 'POST':
        text = request.form['content']

        if not text:
            flash('Please enter text')
        else:
            res = sent.main(text)
            return render_template('sentiment.html', result=res)

    return render_template('sentiment.html')

if __name__ == '__main__':
    port= int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
