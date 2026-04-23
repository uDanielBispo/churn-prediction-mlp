# logger.py — fábrica de loggers com formato estruturado para toda a API.
#
# - logging permite filtrar por nível, redirecionar para arquivos,
#   integrar com ferramentas de observbailidade.
#
# Formato adotado:
#   2026-04-23 17:30:00 | INFO     | src.api.routes | mensagem aqui

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Retorna um logger configurado com formato estruturado.

    O parâmetro 'name' deve ser __name__ do módulo que chama esta função.
    Isso garante que cada log indique de onde veio, facilitando o diagnóstico.

    Exemplo de uso em outro módulo:
        from src.api.core.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Mensagem de exemplo")

    A verificação 'if not logger.handlers' evita adicionar múltiplos handlers
    quando o módulo é importado mais de uma vez (o que acontece no uvicorn
    com --reload ativado).
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
