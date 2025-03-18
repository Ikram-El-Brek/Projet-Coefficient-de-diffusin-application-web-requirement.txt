from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np

app = Flask(__name__)
app.secret_key = 'diffusion_calculator_secret_key'

def compute_diffusion_coefficient(D_AB_0, D_BA_0, x_A, lambda_A, lambda_B, q_A, q_B, theta_A, theta_B, theta_BA, theta_AB, theta_AA, theta_BB, tau_AB, tau_BA, D_exp):
    x_B = 1 - x_A
    phi_A = (x_A * lambda_A) / ((x_A * lambda_A) + (x_B * lambda_B))
    phi_B = (x_B * lambda_B) / ((x_A * lambda_A) + (x_B * lambda_B))

    term1 = x_B * np.log(D_AB_0) + x_A * np.log(D_BA_0) + 2 * (x_A * np.log(x_A / phi_A) + x_B * np.log(x_B / phi_B))
    term2 = 2 * x_A * x_B * ((phi_A / x_A) * (1 - (lambda_A / lambda_B)) + (phi_B / x_B) * (1 - (lambda_B / lambda_A)))
    term3 = (x_B * q_A) * ((1 - theta_BA**2) * np.log(tau_BA) + (1 - theta_BB**2) * tau_AB * np.log(tau_AB))
    term4 = (x_A * q_B) * ((1 - theta_AB**2) * np.log(tau_AB) + (1 - theta_AA**2) * tau_BA * np.log(tau_BA))

    ln_D_AB = term1 + term2 + term3 + term4
    D_AB = np.exp(ln_D_AB)
    error = abs((D_AB - D_exp)) / D_exp * 100

    return D_AB, error

@app.route('/1')
def index():
    return render_template('1.html')

@app.route('/2')
def calculator():
    # Default values for the form
    default_values = {
        'D_AB_0': 2.1e-5,
        'D_BA_0': 2.67e-5,
        'x_A': 0.25,
        'lambda_A': 1.127,
        'lambda_B': 0.973,
        'q_A': 1.432,
        'q_B': 1.4,
        'theta_A': 0.254,
        'theta_B': 0.721,
        'theta_BA': 0.612,
        'theta_AB': 0.261,
        'theta_AA': 0.388,
        'theta_BB': 0.739,
        'tau_AB': 1.0326,
        'tau_BA': 0.5383,
        'D_exp': 1.3296e-5,
    }
    return render_template('2.html', values=default_values)

@app.route('/3', methods=['POST'])
def result():
    if request.method == 'POST':
        # Get form data
        D_AB_0 = float(request.form.get('D_AB_0'))
        D_BA_0 = float(request.form.get('D_BA_0'))
        x_A = float(request.form.get('x_A'))
        lambda_A = float(request.form.get('lambda_A'))
        lambda_B = float(request.form.get('lambda_B'))
        q_A = float(request.form.get('q_A'))
        q_B = float(request.form.get('q_B'))
        theta_A = float(request.form.get('theta_A'))
        theta_B = float(request.form.get('theta_B'))
        theta_BA = float(request.form.get('theta_BA'))
        theta_AB = float(request.form.get('theta_AB'))
        theta_AA = float(request.form.get('theta_AA'))
        theta_BB = float(request.form.get('theta_BB'))
        tau_AB = float(request.form.get('tau_AB'))
        tau_BA = float(request.form.get('tau_BA'))
        D_exp = float(request.form.get('D_exp'))
        
        # Calculate diffusion coefficient
        D_AB, error = compute_diffusion_coefficient(
            D_AB_0, D_BA_0, x_A, lambda_A, lambda_B, q_A, q_B, 
            theta_A, theta_B, theta_BA, theta_AB, theta_AA, theta_BB, 
            tau_AB, tau_BA, D_exp
        )
        
        # Store results in session
        session['D_AB'] = f"{D_AB:.4e}"
        session['error'] = f"{error:.2f}"
        
        return render_template('3.html', D_AB=D_AB, error=error)

if __name__ == '__main__':
    import webbrowser
    webbrowser.open('http://localhost:5000/1')
    app.run()
