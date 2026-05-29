# 🧠 Multimodal RAG — Assistant IA

Bienvenue dans votre assistant RAG multimodal de niveau production !

## Ce que vous pouvez faire

- **💬 Posez des questions** sur vos documents indexés
- **📎 Uploadez des fichiers** (PDF, PNG, JPG) pour les indexer en temps réel
- **⚡ Profitez du cache sémantique** — les questions similaires reçoivent une réponse instantanée

## Fonctionnement

1. Vos documents sont découpés en **chunks texte + images**
2. Chaque chunk est encodé avec **BGE-M3** (texte) ou **CLIP** (images)
3. Les vecteurs sont stockés dans **Milvus** avec index HNSW
4. À chaque question, une **recherche hybride** dense + sparse est effectuée
5. Un **cross-encoder** rerank les résultats
6. **GPT-4o** génère une réponse en citant ses sources

## Raccourcis

| Action | Description |
|--------|-------------|
| Glisser-déposer un fichier | Ingérer et indexer immédiatement |
| `@reset` | Réinitialiser l'historique de la conversation |

---
*Propulsé par Milvus • Redis • CLIP • BGE-M3 • GPT-4o • Chainlit*
