from flask import Flask, render_template, request
import sqlite3 as sql
conn = sql.connect('pendapatan.db')
print ("membuat database baru");
conn.execute('CREATE TABLE IF NOT EXISTS mingguan (id INTEGER NOT NULL PRIMARY KEY, biayaiklan INTEGER, keuntungan INTEGER)');
print ("Tabel berhasil dibuat");
conn.close()
app = Flask(__name__)
@app.route('/')
def home():
   return render_template('home.html')
@app.route('/enternew')
def new_student():
   return render_template('datamingguan.html')
@app.route('/addrec',methods = ['POST', 'GET'])
def addrec():
   if request.method == 'POST':
      try:
         id = request.form['id']
         bi = request.form['bi']
         keu = request.form['keu']
                  
         with sql.connect("pendapatan.db") as con:
            cur = con.cursor()
            cur.execute("INSERT INTO mingguan (id,biayaiklan,keuntungan) VALUES (?,?,?)",(id,bi,keu) )
            con.commit()
            msg = "Record berhasil ditambahkan"
      except:
         con.rollback()
         msg = "error in insert operation"
      
      finally:
         return render_template("result.html",msg = msg)
         con.close()
@app.route('/list')
def list():
   con = sql.connect("pendapatan.db")
   con.row_factory = sql.Row
   
   cur = con.cursor()
   cur.execute("select * from mingguan")
   
   rows = cur.fetchall();
   return render_template("list.html",rows = rows)
if __name__ == '__main__':
   app.run(debug = True)