{ pkgs ? import <nixpkgs> {} }:

{
  devShell = pkgs.mkShell {
    buildInputs = with pkgs; [
      # your python packages here
      python311Packages.pandas
      python311Packages.flask
      python311Packages.pip
      # add other packages from requirements.txt here
      python311Packages.numpy
      python311Packages.matplotlib
      python311Packages.scikit-learn
      python311Packages.flask-sqlalchemy
      python311Packages.flask-session
      python311Packages.statsmodels
      python311Packages.flask-wtf
      python311Packages.python-dotenv
      python311Packages.scipy
      python311Packages.joblib
      python311Packages.seaborn
    ];

    # set environment variables if needed
    # shellHook = ''
    #   export MY_VAR="some_value"
    # '';
  };
}





