# === scripts/deploy.py ===
"""
🚀 Script de déploiement automatisé
Support: Raspberry Pi, Cloud, Local
"""

import argparse
import subprocess
import sys
from pathlib import Path
import logging


def setup_logging():
    """Configuration du logging pour déploiement"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - DEPLOY - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/deployment.log'),
            logging.StreamHandler()
        ]
    )


class SystemDeployer:
    """Déployeur système multi-plateforme"""

    def __init__(self, target: str, config: str = 'production', repo_url: str = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.target = target
        self.config = config
        self.repo_url = repo_url
        self.project_root = Path(__file__).parent.parent

        # Configurations par plateforme
        self.configs = {
            'raspberry_pi': {
                'python_cmd': 'python3',
                'pip_cmd': 'pip3',
                'service_user': 'pi',
                'install_path': '/home/pi/energy-ml-system',
                'systemd_service': True,
                'requirements_extra': ['RPi.GPIO', 'adafruit-circuitpython-dht']
            },
            'cloud': {
                'docker_enabled': True,
                'ssl_enabled': True,
                'load_balancer': True
            },
            'local': {
                'python_cmd': 'python',
                'pip_cmd': 'pip',
                'development_mode': True
            }
        }

    def deploy(self):
        """Déploiement principal"""
        self.logger.info(f"🚀 Démarrage déploiement {self.target} en mode {self.config}")

        try:
            if self.target == 'raspberry_pi':
                self.deploy_raspberry_pi()
            elif self.target == 'cloud':
                self.deploy_cloud()
            elif self.target == 'local':
                self.deploy_local()
            else:
                raise ValueError(f"Plateforme {self.target} non supportée")

            self.logger.info("✅ Déploiement terminé avec succès!")

        except Exception as e:
            self.logger.error(f"❌ Erreur déploiement: {e}")
            raise

    # === Raspberry Pi ===
    def deploy_raspberry_pi(self):
        """Déploiement spécifique Raspberry Pi"""
        self.logger.info("🍓 Déploiement Raspberry Pi...")
        config = self.configs['raspberry_pi']

        # 1. Cloner ou mettre à jour le dépôt Git
        if self.repo_url:
            if not Path(config['install_path']).exists():
                self.run_command(f"git clone {self.repo_url} {config['install_path']}")
            else:
                self.run_command(f"cd {config['install_path']} && git pull origin main")

        # 2. Mise à jour système
        self.run_command('sudo apt update && sudo apt upgrade -y')

        # 3. Installation dépendances système
        system_deps = [
            'python3-pip', 'python3-venv', 'git', 'curl',
            'libhdf5-dev', 'libatlas-base-dev',
            'python3-dev', 'build-essential'
        ]
        self.run_command(f"sudo apt install -y {' '.join(system_deps)}")

        # 4. Création environnement virtuel
        venv_path = Path(config['install_path']) / 'venv'
        self.run_command(f"python3 -m venv {venv_path}")

        # 5. Installation requirements
        pip_cmd = f"{venv_path}/bin/pip"
        self.run_command(f"{pip_cmd} install --upgrade pip")
        self.run_command(f"{pip_cmd} install -r {config['install_path']}/requirements.txt")

        # 6. Installation dépendances spécifiques
        for extra_dep in config['requirements_extra']:
            self.run_command(f"{pip_cmd} install {extra_dep}")

        # 7. Configuration service systemd
        if config['systemd_service']:
            self.create_systemd_service(config['install_path'], config['service_user'])

        # 8. Permissions
        self.run_command(f"sudo chown -R {config['service_user']}:{config['service_user']} {config['install_path']}")

    def create_systemd_service(self, install_path, user):
        """Créer un service systemd pour exécuter l’application"""
        service_file = f"""
        [Unit]
        Description=Energy ML System Service
        After=network.target

        [Service]
        User={user}
        WorkingDirectory={install_path}
        ExecStart={install_path}/venv/bin/python main.py dashboard
        Restart=always

        [Install]
        WantedBy=multi-user.target
        """

        service_path = "/etc/systemd/system/energy-ml-system.service"
        with open("energy-ml-system.service", "w") as f:
            f.write(service_file)

        self.run_command(f"sudo mv energy-ml-system.service {service_path}")
        self.run_command("sudo systemctl daemon-reload")
        self.run_command("sudo systemctl enable energy-ml-system")

    # === Cloud ===
    def deploy_cloud(self):
        """Déploiement cloud (Docker)"""
        self.logger.info("☁️ Déploiement Cloud...")

        self.create_dockerfile()
        self.create_docker_compose()

        self.run_command('docker-compose build')
        self.run_command('docker-compose up -d')

        if self.configs['cloud'].get('ssl_enabled'):
            self.setup_ssl()

    def create_dockerfile(self):
        """Créer un Dockerfile basique"""
        dockerfile = """
        FROM python:3.10-slim
        WORKDIR /app
        COPY . .
        RUN pip install --no-cache-dir -r requirements.txt
        CMD ["python", "main.py", "dashboard"]
        """
        with open("Dockerfile", "w") as f:
            f.write(dockerfile)

    def create_docker_compose(self):
        """Créer un docker-compose.yml basique"""
        compose = """
        version: '3.8'
        services:
          energy-ml:
            build: .
            ports:
              - "8501:8501"
            restart: always
        """
        with open("docker-compose.yml", "w") as f:
            f.write(compose)

    def setup_ssl(self):
        self.logger.info("🔒 Configuration SSL (à compléter selon votre domaine et certbot)")

    # === Local ===
    def deploy_local(self):
        """Déploiement local développement"""
        self.logger.info("💻 Déploiement Local...")
        self.run_command('pip install -r requirements.txt')

    # === Utilitaires ===
    def run_command(self, command: str):
        """Exécuter commande système avec logging"""
        self.logger.info(f"➡️ Exécution: {command}")
        try:
            result = subprocess.run(
                command, shell=True, check=True,
                capture_output=True, text=True
            )
            if result.stdout:
                self.logger.debug(result.stdout.strip())
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Erreur commande: {e.stderr}")
            raise


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Script de déploiement Energy ML System')
    parser.add_argument('--target',
                        choices=['raspberry_pi', 'cloud', 'local'],
                        required=True,
                        help='Plateforme de déploiement')
    parser.add_argument('--config',
                        choices=['dev', 'staging', 'production'],
                        default='production',
                        help="Configuration d'environnement")
    parser.add_argument('--repo',
                        help="URL du dépôt GitHub à déployer")
    args = parser.parse_args()

    deployer = SystemDeployer(args.target, args.config, args.repo)
    deployer.deploy()

    print(f"\n🎉 Déploiement {args.target} terminé avec succès!")
    if args.target == 'raspberry_pi':
        print("➡️ sudo systemctl start energy-ml-system")
        print("➡️ journalctl -u energy-ml-system -f")
        print("➡️ http://ip_raspberry:8501")
    elif args.target == 'cloud':
        print("➡️ docker-compose ps")
        print("➡️ http://votre-domaine:8501")
    elif args.target == 'local':
        print("➡️ python main.py dashboard")
        print("➡️ http://localhost:8501")


if __name__ == "__main__":
    sys.exit(main())
