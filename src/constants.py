import os
import random
import string
import uuid

BASE_DOMAINS = [
    "example.com",
    "test-site.org",
    "company.io",
    "webapp.dev",
    "corporate.net",
    "app.example.com",
    "api.test-site.org",
    "acme.corp",
    "internal.example.com",
    "dev.company.io",
    "staging.webapp.dev",
    "partner.corporate.net",
    "support.example.com",
    "blog.test-site.org",
    "shop.company.io",
    "my.acme.corp",
]

ADMIN_PATH_SEGMENTS = [
    "admin",
    "adm",
    "admincp",
    "modcp",
    "control",
    "panel",
    "acp",
    "dashboard",
    "manage",
    "management",
    "settings",
    "config",
    "root",
]

API_PATH_SEGMENTS = [
    "api",
    "rest",
    "graphql",
    "rpc",
    "service",
    "v1",
    "v2",
    "v3",
    "latest",
]

AUTH_PATH_SEGMENTS = [
    "auth",
    "account",
    "login",
    "logout",
    "register",
    "signin",
    "signup",
    "password",
    "profile",
    "user",
    "oauth",
    "oidc",
    "saml",
    "sso",
    "connect",
]

FILE_PATH_SEGMENTS = [
    "files",
    "uploads",
    "downloads",
    "static",
    "assets",
    "media",
    "content",
    "images",
    "img",
    "js",
    "css",
    "scripts",
    "styles",
    "documents",
    "attachments",
    "export",
    "import",
    "backup",
    "temp",
    "storage",
    "avatar",
    "common",
    "common-v2",
]

COMMON_PATH_SEGMENTS = [
    "search",
    "product",
    "category",
    "item",
    "order",
    "cart",
    "checkout",
    "user",
    "profile",
    "setting",
    "report",
    "data",
    "list",
    "detail",
    "view",
    "edit",
    "create",
    "update",
    "delete",
    "post",
    "article",
    "page",
    "event",
    "news",
    "blog",
    "support",
    "help",
    "faq",
    "contact",
    "about",
]

INTERESTING_FILE_EXTENSIONS = [
    ".sql",
    ".bak",
    ".backup",
    ".env",
    ".config",
    ".conf",
    ".yml",
    ".yaml",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".cer",
    ".crt",
    ".ini",
    ".log",
    ".txt",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".rar",
    ".jar",
    ".war",
    ".ear",
    ".php",
    ".phtml",
    ".php3",
    ".php4",
    ".php5",
    ".phps",
    ".asp",
    ".aspx",
    ".jsp",
    ".jspx",
    ".cfm",
    ".cgi",
    ".pl",
    ".py",
    ".rb",
    ".sh",
    ".bat",
    ".ps1",
    ".bak~",
    ".old",
    ".swp",
    # Additional extensions from security reports
    ".git",
    ".svn",
    ".hg",
    ".idea",
    ".htaccess",
    ".htpasswd",
    ".DS_Store",
    ".dockerignore",
    ".npmrc",
    ".yarnrc",
    ".editorconfig",
    ".jks",
    ".keystore",
    ".pem.bak",
    ".key.old",
    ".ssh",
    ".history",
    ".bash_history",
    ".zsh_history",
    ".json.bak",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".mdb",
    ".properties",
    ".settings",
    ".inc",
    ".config.js",
    ".config.php",
    ".config.json",
    ".secrets.js",
    ".secrets.yml",
]

COMMON_FILE_EXTENSIONS = [
    ".html",
    ".htm",
    ".css",
    ".js",
    ".json",
    ".xml",
    ".csv",
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".mp4",
    ".mp3",
    ".webm",
    ".avi",
    ".mov",
]

INTERESTING_PARAMS = [
    # Command injection
    "cmd",
    "exec",
    "command",
    "run",
    "system",
    "shell",
    "execute",
    "ping",
    "query",
    "expr",
    # File operations
    "file",
    "path",
    "filename",
    "upload",
    "download",
    "document",
    "attachment",
    "include",
    "require",
    "import",
    "export",
    "load",
    "save",
    "open",
    "read",
    "dir",
    "folder",
    # Database/SQLi
    "id",
    "user_id",
    "order_id",
    "item_id",
    "product_id",
    "category_id",
    "search",
    "query",
    "sql",
    "filter",
    "where",
    "select",
    "delete",
    "update",
    "insert",
    "union",
    "table",
    "column",
    # Authentication/Session
    "username",
    "user",
    "pass",
    "password",
    "token",
    "auth",
    "key",
    "apikey",
    "session",
    "sessionid",
    "cookie",
    "jwt",
    "secret",
    "hash",
    "salt",
    "otp",
    "pin",
    "credential",
    "access_token",
    # Application flow/Redirects/SSRF
    "redirect",
    "redirect_uri",
    "return",
    "returnUrl",
    "return_url",
    "return_to",
    "back",
    "next",
    "url",
    "target",
    "dest",
    "destination",
    "forward",
    "goto",
    "continue",
    "uri",
    "host",
    "port",
    "r",
    "success_url",
    "error_url",
    "cancel_url",
    "callback",
    "webhook",
    "proxy",
    "fetch",
    "out",
    # Template Injection (SSTI)
    "template",
    "view",
    "layout",
    "name",
    "id",
    "data",
    "content",
    "preview",
    "format",
    "render",
    # XXE
    "xml",
    "data",
    "content",
    "input",
    "request",
    "doc",
    # Debug/Testing/Admin
    "debug",
    "test",
    "dev",
    "beta",
    "sandbox",
    "mock",
    "fake",
    "dummy",
    "enable",
    "disable",
    "admin",
    "root",
    "manage",
    "control",
    "grant",
    "revoke",
    "allow",
    "deny",
    "action",
    "mode",
    # Additional params from security reports
    "code",
    "signature",
    "s3_url",
    "endpoint",
    "remote",
    "domain",
    "callback_url",
    "api_url",
    "csp-report",
    "document_domain",
    "jsonp",
    "resource",
    "method",
    "eval",
    "state",
    "code_challenge",
    "access",
    "scope",
    "display",
    "response_type",
    "client_secret",
    "grant_type",
    "assertion",
    "original_url",
    "image_url",
    "avatar_url",
    "profilePhoto",
    "json",
    "jsonp_callback",
    "api_key",
    "auth_token",
    "oauth_token",
    "auth_signature",
    "saml_response",
    "wresult",
    "RelayState",
    "function",
    "func",
    "customFunction",
    "generate",
    "template_url",
    "theme_url",
    "account_id",
    "group_id",
    "organization_id",
    "team_id",
    "role_id",
    "customer_id",
    "shop_id",
    "site_id",
    "tenant_id",
    "referrer_policy",
]

BORING_PARAMS = [
    "page",
    "limit",
    "offset",
    "size",
    "sort",
    "order",
    "direction",
    "filter",
    "q",
    "search",
    "keyword",
    "category",
    "tag",
    "type",
    "kind",
    "format",
    "view",
    "layout",
    "style",
    "theme",
    "color",
    "lang",
    "locale",
    "currency",
    "country",
    "region",
    "zone",
    "timestamp",
    "nonce",
    "csrf_token",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
    "referrer",
    "gclid",
    "fbclid",
    "msclkid",
    "campaignid",
    "adgroupid",
    "adid",
    "kwid",
    "matchtype",
    "network",
    "device",
    "creative",
    "placement",
    "targetid",
]

URL_VALUE_PARAMS = [
    "redirect",
    "redirect_uri",
    "return",
    "returnUrl",
    "return_url",
    "return_to",
    "back",
    "next",
    "url",
    "target",
    "dest",
    "destination",
    "forward",
    "goto",
    "continue",
    "uri",
    "host",
    "r",
    "callback",
    "webhook",
    "proxy",
    "fetch",
    "out",
    "success_url",
    "error_url",
    "cancel_url",
    "img_src",
    "script_src",
    "file_url",
    "doc_url",
    # Additional URL params from security reports
    "callback_url",
    "api_url",
    "avatar_url",
    "image_url",
    "document_url",
    "photo_url",
    "preview_url",
    "remote_url",
    "webhook_url",
    "download_url",
    "upload_url",
    "endpoint_url",
    "service_url",
    "media_url",
    "asset_url",
    "attachment_url",
    "resource_url",
    "referrer",
    "template_url",
    "cdn_url",
    "origin",
    "from_url",
    "original_url",
    "sender_url",
    "source_url",
    "external_url",
]


def generate_id(style: str = "int") -> str:
    """Generate a random identifier based on the specified style.

    Args:
        style: The style of ID to generate. Options are:
            - "uuid": Generate a UUID4 string
            - "hex": Generate a random 8-byte hex string
            - "alnum": Generate a random alphanumeric string (8-16 chars)
            - "int": Generate a random integer (1-99999)

    Returns:
        A string containing the generated identifier.
    """
    if style == "uuid":
        return str(uuid.uuid4())
    elif style == "hex":
        return os.urandom(8).hex()
    elif style == "alnum":
        return "".join(
            random.choices(
                string.ascii_lowercase + string.digits, k=random.randint(8, 16)
            )
        )
    else:
        return str(random.randint(1, 99999))


def generate_random_string(length: int = 10) -> str:
    """Generate a random alphanumeric string of specified length.

    Args:
        length: The desired length of the random string. Defaults to 10.

    Returns:
        A string containing random alphanumeric characters.
    """
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


COMMON_VALUES = [
    "true",
    "false",
    "1",
    "0",
    "yes",
    "no",
    "on",
    "off",
    "null",
    "undefined",
    "test",
    "admin",
    "user",
    "guest",
    "default",
    "en",
    "us",
    "gb",
    "all",
    "none",
    "list",
    "grid",
    "asc",
    "desc",
    "name",
    "date",
    "price",
    "image/png",
    "application/json",
    "text/html",
] + [str(i) for i in range(1, 20)]

INTERESTING_PATHS = [
    # Admin interfaces
    "/admin",
    "/administrator",
    "/wp-admin",
    "/dashboard",
    "/console",
    "/manage",
    "/management",
    "/control",
    "/panel",
    "/adm",
    "/backoffice",
    "/admincp",
    "/modcp",
    "/acp",
    "/settings",
    "/config",
    "/root",
    # Authentication endpoints
    "/login",
    "/signin",
    "/signup",
    "/register",
    "/auth",
    "/account",
    "/sso",
    "/oauth",
    "/oidc",
    "/saml",
    "/connect",
    "/password-reset",
    "/forgot-password",
    "/change-password",
    "/profile",
    "/user",
    "/.well-known/openid-configuration",
    # API endpoints
    "/api",
    "/api/v1",
    "/api/v2",
    "/graphql",
    "/rest",
    "/rpc",
    "/service",
    "/api/users",
    "/api/data",
    "/api/admin",
    "/api/config",
    "/_api",
    # File related
    "/download",
    "/upload",
    "/file",
    "/files",
    "/documents",
    "/attachments",
    "/media",
    "/imports",
    "/exports",
    "/backup",
    "/backups",
    "/static",
    "/assets",
    "/content",
    "/storage",
    "/temp",
    # Database related / Data Exposure
    "/sql",
    "/query",
    "/db",
    "/database",
    "/data",
    "/export",
    "/search",
    "/list",
    "/report",
    "/metrics",
    # Common vulnerable files/paths
    "/config",
    "/settings",
    "/system",
    "/debug",
    "/test",
    "/dev",
    "/env",
    "/logs",
    "/phpinfo.php",
    "/info.php",
    "/status",
    "/health",
    "/server-status",
    "/server-info",
    "/.git",
    "/.svn",
    "/.hg",
    "/.env",
    "/.aws",
    "/.ssh",
    "/web.config",
    "/appsettings.json",
    "/docker-compose.yml",
    "/Makefile",
    "/package.json",
    "/composer.json",
    "/wp-config.php",
    # Execute/Tools
    "/execute",
    "/tools",
    "/shell",
    "/ping",
    "/run",
    "/command",
    # Internal services / Other
    "/internal",
    "/private",
    "/staging",
    "/dev",
    "/test",
    "/beta",
    "/local",
    "/jenkins",
    "/jira",
    "/gitlab",
    "/kibana",
    "/grafana",
    "/prometheus",
    "/swagger",
    "/proxy",
    "/redirect",
    "/callback",
    "/webhook",
    # Additional paths from security reports
    "/.git/config",
    "/.git/HEAD",
    "/actuator",
    "/actuator/env",
    "/actuator/health",
    "/actuator/metrics",
    "/api/graphiql",
    "/api/swagger",
    "/api/swagger-ui",
    "/api/swagger-ui.html",
    "/api/v1/swagger-ui",
    "/api/v1/users/bulk",
    "/api/admin/auth",
    "/.well-known/security.txt",
    "/cgi-bin/status",
    "/telescope",
    "/laravel-websockets",
    "/storage/logs",
    "/vendor/phpunit/phpunit",
    "/wp-content/debug.log",
    "/rails/info/properties",
    "/rails/info/routes",
    "/node_modules",
    "/site.xml",
    "/server.xml",
    "/wp-json/wp/v2/users",
    "/struts/utils.js",
    "/api/secrets",
    "/api/keys",
    "/api/tokens",
    "/api/credentials",
    "/__debug__/",
    "/webadmin",
    "/rails/db",
    "/telescope/requests",
    "/_profiler/",
    "/phpunit/src/Util/PHP/eval-stdin.php",
    "/solr/admin/",
    "/apc.php",
    "/memcached.php",
    "/adminer.php",
    "/elmah.axd",
    "/site/browserconfig.xml",
    "/user/login.json",
]

NORMAL_PATHS = [
    "/",
    "/home",
    "/about",
    "/contact",
    "/content",
    "/landing",
    "/images",
    "/docs",
    "/contact-us",
    "/products",
    "/services",
    "/blog",
    "/news",
    "/faq",
    "/articles",
    "/help",
    "/support",
    "/terms",
    "/privacy",
    "/sitemap",
    "/search",
    "/category",
    "/tag",
    "/page",
    "/post",
    "/article",
    "/resource",
    "/media",
    "/asset",
    "/static",
    "/css",
    "/js",
    "/img",
    "/images",
    "/fonts",
    "/icons",
    "/themes",
    "/styles",
    "/scripts",
    "/public",
    "/videos",
    "/audio",
    "/portfolio",
    "/gallery",
    "/legal",
    "/careers",
    "/jobs",
    "/press",
    "/events",
]

NORMAL_PARAMS = [
    "page",
    "limit",
    "offset",
    "size",
    "sort",
    "order",
    "direction",
    "filter",
    "q",
    "query",
    "search",
    "keyword",
    "category",
    "tag",
    "type",
    "kind",
    "format",
    "view",
    "layout",
    "style",
    "theme",
    "color",
    "lang",
    "locale",
    "currency",
    "country",
    "region",
    "zone",
    "timestamp",
    "nonce",
    "csrf_token",
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "source",
    "referrer",
    "gclid",
    "fbclid",
    "msclkid",
    "campaignid",
    "adgroupid",
    "adid",
    "kwid",
    "matchtype",
    "network",
    "device",
    "creative",
    "placement",
    "targetid",
    "item",
    "product",
    "id",
    "article_id",
    "post_id",
]

LOW_PRIORITY_PATHS = [
    "/avatar/",
    "/captcha/",
    "/images/",
    "/img/",
    "/static/",
    "/assets/",
    "/css/",
    "/fonts/",
    "/svg/",
    "/common-v2/",
    "/CaCertificates/",
    "/Files/",
    "/FileSys/",
]

LOCALIZATION_PATHS = [
    "/l10n/",
    "/localization/",
    "/i18n/",
    "/lang/",
    "/locale/",
]

STATIC_INDICATORS_FOR_FEATURE_EXTRACTION = [
    "/icons/",
    "/images/",
    "/img/",
    "/static/",
    "/assets/",
    "/css/",
    "/fonts/",
    "/svg/",
    "/common-v2/",
    "/common/",
    "/avatar/",
    "/styles/",
    "/dist/",
    "/build/",
    "/public/",
    "/cdn/",
    "/themes/",
]

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".ico",
    ".bmp",
    ".webp",
    ".logo",
]

MEDIA_EXTENSIONS = [
    ".mp3",
    ".mp4",
    ".wav",
    ".ogg",
    ".avi",
    ".webm",
    ".mov",
    ".wmv",
    ".flv",
    ".mkv",
    ".m4a",
    ".m4v",
    ".3gp",
]

FILES_TO_PENALIZE = {
    ".js",
    ".css",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".svg",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pdf",
    ".docx",
    ".doc",
    ".xlsx",
    ".xls",
    ".pptx",
    ".ppt",
    ".zip",
    ".tar",
    ".gz",
    ".mp4",
    ".webm",
    ".ogg",
    ".mp3",
    ".wav",
    ".xml",
    ".txt",
    ".log",
    ".json",
    ".csv",
}
