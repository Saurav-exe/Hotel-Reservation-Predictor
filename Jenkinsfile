pipeline {
  agent any

  environment {
    VENV_DIR = 'venv'
    GCP_PROJECT="ancient-kayak-463903-b4"
    GCLOUD_PATH="/var/jenkins_home/google-cloud-sdk/bin"
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

    stage('Building and pushing docker img to GCR') {
      steps {
        withCredential([file(credentialsId:'GCPKEY',variable:"GOOGLE_APPLICATION_CREDENTIALS")]){
          script{
            echo'Building and pushing docker img to GCR'
            sh '''
            export PATH=$PATH:$(GCLOUD_PATH)

            gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

            gcloud config set project ${GCP_PROJECT}

            gcloud auth configure-docker --quite

            docker build -t gcr.io/${GCP_PROJECT}/ml-project:latest .

            docker push gcr.io/${GCP_PROJECT}/ml-project:latest


            '''
          }
        }


        }
      }
    } 
  }

