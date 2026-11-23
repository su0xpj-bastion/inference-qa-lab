// Same stages as .github/workflows/ci.yml, for Jenkins agents with Docker + kind.
pipeline {
  agent { label 'docker' }
  options { timeout(time: 40, unit: 'MINUTES'); disableConcurrentBuilds(abortPrevious: true) }
  environment {
    CLUSTER = "inference-${env.BUILD_NUMBER}"
    INGRESS_URL = 'http://localhost:8080'
  }
  stages {
    stage('Setup') {
      steps { sh 'python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt' }
    }
    stage('Unit + integration') {
      parallel {
        stage('python') { steps { sh '.venv/bin/python -m pytest --junitxml=reports/junit.xml' } }
        stage('go')     { steps { dir('tools/lbprobe') { sh 'go vet ./... && go test ./...' } } }
      }
    }
    stage('Smoke') { steps { sh 'scripts/smoke.sh' } }
    stage('Cluster') {
      steps {
        sh 'kind create cluster --name $CLUSTER --config deploy/kind/kind-config.yaml'
        sh 'scripts/kind_deploy.sh $CLUSTER'
        sh '.venv/bin/python -m pytest -m cluster tests/cluster -o addopts="" --junitxml=reports/junit-cluster.xml'
      }
      post {
        failure { sh 'scripts/collect_diagnostics.sh || true' }
        always  { sh 'kind delete cluster --name $CLUSTER || true' }
      }
    }
  }
  post {
    always {
      junit allowEmptyResults: true, testResults: 'reports/junit*.xml'
      archiveArtifacts allowEmptyArchive: true, artifacts: 'reports/**'
    }
  }
}
