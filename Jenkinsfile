pipeline{
  agent any

  stages{
    stage('Clone Repository'){
      steps{
        script{
          echo 'Cloning repository...'
          checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'Github-token', url: 'https://github.com/Saurav-exe/Hotel-Reservation-Predictor.git']])
        }
  }
}
}
}