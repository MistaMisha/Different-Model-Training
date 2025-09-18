from flask import Blueprint, request, session, redirect, url_for, render_template, flash
import json, hashlib

auth_bp = Blueprint("auth", __name__)

def load_users():
    with open("users_hashed.json") as f:
        return json.load(f)

def hash_pw(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        users = load_users()
        for u in users:
            if u["username"] == username and u["password"] == hash_pw(password):
                session["user"] = {
                    "username": u["username"],
                    "role": u["role"],
                    "outlet_name": u.get("outlet_name")
                }
                return redirect(url_for("show_results"))
        flash("Invalid credentials")
    return render_template("login.html")

@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("auth.login"))
