pipeline {
  agent any

  environment {
    VENV_DIR = 'venv'
  }

  stages {
    stage('Clone Repository') {
      steps {
        script {
          echo 'Cloning repository...'
          checkout scmGit(
            branches: [[name: '*/main']],
            extensions: [],
            userRemoteConfigs: [[
              credentialsId: 'Github-token',
              url: 'https://github.com/Saurav-exe/Hotel-Reservation-Predictor.git'
            ]]
          )
        }
      }
    }

    stage('Setup Virtual Environment and Install Dependencies') {
      steps {
        script {
          echo 'Setting up Virtual Environment and Installing Dependencies...'
          sh '''
          python3 -m venv ${VENV_DIR}
          . ${VENV_DIR}/bin/activate
          pip install --upgrade pip
          if [ -f requirements.txt ]; then
          pip install -r requirements.txt
          fi
          '''

        }
      }
    }
  }
}
