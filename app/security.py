"""密码哈希与 JWT 签发/校验（标准库 scrypt + PyJWT）。

为什么这样选型：
- 密码哈希用 hashlib.scrypt：标准库自带，无需 bcrypt/argon2 依赖；
  scrypt 带盐、可调成本（n=2^14），适合单机演示；
- JWT 用 PyJWT（HS256）：无状态 token，sub 放用户 ID、role 放角色，
  过期时间 exp 强制登录态过期；密钥只存服务端，客户端无法伪造。

⚠️ 生产注意：HS256 密钥要放密钥管理（KMS/环境变量注入），
且登录接口要加限流防爆破（本项目还缺，见审查记录 P0 清单）。
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import time

import jwt

from . import config

JWT_ALGORITHM = "HS256"
JWT_TTL_SECONDS = int(os.getenv("JWT_TTL_SECONDS", "43200"))  # 12 小时

# scrypt 参数：n 越大越慢（抗暴力），n=2^14 是交互式登录的常用档位
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1


def get_jwt_secret() -> str:
    """JWT 签名密钥：优先环境变量 JWT_SECRET；没配就自动生成并落盘持久化。

    自动生成到 DATA_DIR/.jwt_secret（git 已忽略）：
    保证重启后 token 不失效，演示环境零配置也能跑。
    """
    if config.JWT_SECRET:
        return config.JWT_SECRET
    path = config.DATA_DIR / ".jwt_secret"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def hash_password(password: str) -> str:
    """生成「算法$盐$摘要」格式的密码哈希，盐随机、可自校验。"""
    salt = os.urandom(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32
    )
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码。任何异常一律返回 False（不区分「格式坏」与「密码错」）。"""
    try:
        _, salt_hex, digest_hex = stored.split("$")
        digest = hashlib.scrypt(
            password.encode("utf-8"),
            salt=bytes.fromhex(salt_hex),
            n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32,
        )
        # compare_digest：常数时间比较，防时序侧信道
        return hmac.compare_digest(digest.hex(), digest_hex)
    except Exception:  # noqa: BLE001
        return False


def issue_token(user_id: int, role: str) -> str:
    """签发 JWT。payload 只放最小必要信息（sub/role/iat/exp）。"""
    now = int(time.time())
    return jwt.encode(
        {"sub": str(user_id), "role": role, "iat": now, "exp": now + JWT_TTL_SECONDS},
        get_jwt_secret(),
        algorithm=JWT_ALGORITHM,
    )


def decode_token(token: str) -> dict | None:
    """校验签名与过期时间。无效/过期一律返回 None。"""
    try:
        return jwt.decode(token, get_jwt_secret(), algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        return None
