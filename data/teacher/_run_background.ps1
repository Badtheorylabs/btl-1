$env:BTL_TEACHER_API_KEY = "sk-b644b3c0b0384b5e910d46cca6e983b8"
$env:BTL_TEACHER_MODEL = "deepseek-v4-flash"
cd "C:\Users\pc\Downloads\btl\btl-1\data\teacher"
node run.mjs --concurrency=20 *> "C:\Users\pc\Downloads\btl\btl-1\data\teacher\run.log"
