# WSTAD aplicado a tráfego HTTP

Este repositório adapta a ideia do artigo de detecção de anomalias fracamente supervisionada para distinguir sequências de requisições HTTP feitas por humanos e por bots.

## Mapeamento do problema

- Vídeo no artigo: sessão ou janela temporal de tráfego HTTP.
- Segmento/frame: uma requisição HTTP individual.
- Bag anômala: sessão/janela que contém comportamento de bot.
- Bag normal: sessão/janela humana.
- Score temporal: probabilidade de cada requisição da sessão ser bot.

O formato esperado de cada evento é:

```python
{
    "datetime": "2026-05-03T12:00:00",
    "ip": "203.0.113.10",
    "headers": {"user-agent": "..."},
    "request": {"q": "..."},
    "decision": "bot"  # ou "bots", "human", "humano", "unsafe"
}
```

## Pré-processamento

Use `preprocess_http_events` para transformar uma lista de eventos em tensores:

```python
from http_preprocessing import preprocess_http_events

X, y, mask, session_ids = preprocess_http_events(events, feature_dim=64, max_len=128)
```

Saídas:

- `X`: tensor `(batch, tempo, features)`.
- `y`: rótulo fraco por sessão, `1` para bot e `0` para humano.
- `mask`: posições válidas quando sessões têm tamanhos diferentes.
- `session_ids`: ids usados para agrupar as requisições.

Por padrão, a sessão é inferida pelo campo `ip` do evento. Se `ip` não existir, o preprocessador tenta `remote_addr`, `client_ip`, `x-forwarded-for` e `x-real-ip`.

Se quiser agrupar por IP e janela temporal, passe uma função customizada em `session_key_fn`. Isso evita sessões muito longas quando um mesmo IP aparece durante horas ou dias.

```python
from http_preprocessing import extract_ip, parse_datetime

def ip_hour_key(event):
    dt = parse_datetime(event["datetime"])
    return f"{extract_ip(event)}|{dt:%Y-%m-%d-%H}"

X, y, mask, session_ids = preprocess_http_events(
    events,
    feature_dim=64,
    max_len=128,
    session_key_fn=ip_hour_key,
)
```

## Features textuais

O preprocessador atual usa hashing de tokens de cabeçalhos e parâmetros, que é uma baseline leve e rápida. Para capturar semântica, substitua ou complemente `event_to_feature` com embeddings textuais.

Uma representação simples para cada requisição é serializar os campos assim:

```text
headers: user-agent=... accept=... referer=...
request: q=... page=... utm_source=...
```

Depois, gere um embedding com um modelo de linguagem e concatene com features numéricas como delta temporal, quantidade de headers, quantidade de parâmetros, tamanho do user-agent e entropias. Em produção, vale testar três níveis:

- Baseline: hashing + features numéricas, como está agora.
- Semântico local: `sentence-transformers`, por exemplo MiniLM, para embeddings de 384 dimensões.
- Semântico via API/modelo maior: embeddings robustos para valores textuais ruidosos, URLs, user-agents e combinações raras.

`http_preprocessing.event_to_text(event)` já cria uma versão textual da requisição para esse tipo de encoder.

## Recomendações após diagnóstico

Com os dados TikTok de exemplo, `--session-mode ip` gerou muitas sessões de tamanho 1. Isso enfraquece a adaptação direta do artigo, porque WAGCN e MIL precisam de uma sequência de instâncias dentro da bag.

Use o teste assim para diagnosticar o agrupamento por IP:

```powershell
.\.venv\Scripts\python main.py --limit-per-file 1000 --session-mode ip
```

Se a maioria das sessões continuar com uma única requisição, use bags artificiais rotuladas para treinar a hipótese WSTAD:

```powershell
.\.venv\Scripts\python main.py --limit-per-file 1000 --session-mode label-chunk --chunk-size 32 --max-len 32
```

Interpretação recomendada:

- `ip`: mais próximo da entidade real, mas ruim se cada IP aparece uma vez.
- `ip-hour`: útil quando há tráfego recorrente do mesmo IP ao longo do tempo.
- `ip-user-agent-hour`: mais estrito; reduz mistura de usuários por NAT, mas pode deixar sessões ainda menores.
- `label-chunk`: melhor para validar a arquitetura do artigo em dados tabulares rotulados; não deve ser confundido com uma estratégia final de inferência online.

Como os rótulos são fracamente supervisionados, trate `decision` como rótulo fraco. Pode existir bot em `unsafe` e humano/unsafe em `bot`. Portanto, as métricas semânticas abaixo não devem ser lidas como acurácia final, mas como diagnóstico de separação e ruído.

Para este dataset, o caminho experimental mais honesto é:

1. Rodar uma baseline por requisição, porque o rótulo já existe por evento.
2. Rodar WSTAD com `label-chunk` para testar se a arquitetura aprende separação de bags.
3. Rodar WSTAD por IP apenas se houver sessões com várias requisições por IP.
4. Comparar TikTok, Taboola e Outbrain separadamente e também em treino cruzado, por exemplo treinar em TikTok e testar em Taboola.
5. Evoluir as features textuais para embeddings e comparar contra hashing.

Na parte semântica, evite embeddar tudo de forma ingênua sem limpeza. Separe pelo menos estes canais:

- `user-agent`: família do navegador/app, sistema operacional, webview, app version, indícios de crawler.
- headers de rede/CDN: `cf-*`, `x-forwarded-*`, proxy, hosting e país.
- parâmetros de campanha: `utm_*`, `ttclid`, `ref_id`, `sub*`.
- conteúdo livre: textos de anúncio, busca, nomes de campanha e parâmetros longos.

Uma boa feature final por requisição é:

```text
[embedding_user_agent ; embedding_request_params ; embedding_campaign_text ; numeric_features]
```

Isso tende a ser melhor do que misturar todos os campos em uma única string, porque cada canal tem semântica e ruído diferentes.

## Validação semântica com MiniLM

O modo semântico usa `sentence-transformers/all-MiniLM-L6-v2` para transformar a requisição serializada em embedding de 384 dimensões e concatena 8 features numéricas. Assim, cada requisição vira um vetor de 392 dimensões.

Primeira execução, se o modelo ainda não estiver em cache:

```powershell
.\.venv\Scripts\python main.py --limit-per-file 100 --session-mode label-chunk --chunk-size 20 --max-len 20 --feature-mode minilm
```

Depois que o modelo estiver em cache, prefira modo offline para evitar tentativas de rede:

```powershell
.\.venv\Scripts\python main.py --limit-per-file 1000 --session-mode label-chunk --chunk-size 32 --max-len 32 --feature-mode minilm --minilm-offline
```

Métricas emitidas pelo diagnóstico:

- `semantic centroid cosine unsafe/bot`: similaridade entre centróides dos dois rótulos fracos. Quanto mais perto de 1, mais parecidos globalmente.
- `semantic centroid weak-label agreement`: fração de eventos que ficam mais próximos do centróide do próprio rótulo fraco.
- `semantic top-5 neighbor weak-label agreement`: fração dos 5 vizinhos semânticos mais próximos que compartilham o mesmo rótulo fraco.
- `semantic suspect weak labels unsafe->bot/bot->unsafe`: eventos semanticamente mais próximos do centróide oposto; bons candidatos a ruído de rótulo.

Resultado observado em TikTok com `1000 + 1000`:

```text
semantic centroid cosine unsafe/bot: 0.9774
semantic centroid weak-label agreement: 0.8255
semantic top-5 neighbor weak-label agreement: 0.8483
semantic suspect weak labels unsafe->bot/bot->unsafe: 177/172
X shape: (64, 32, 392)
```

Leitura recomendada: os centróides globais são muito próximos, então a fronteira não é trivial. Mesmo assim, os vizinhos locais concordam com o rótulo fraco em torno de 85%, indicando que há sinal semântico útil. Os `177 + 172` suspeitos reforçam a hipótese de rótulo fraco/ruidoso.

## Modelo

`BotDetectionNet` recebe `X` e retorna:

- `scores`: score por requisição, usado pela perda MIL.
- `features`: embedding por requisição, usado pela perda contrastiva.

Exemplo:

```python
from crossbatch_memory_banck import BotDetectionNet, CrossBatchMemoryBank
from mil_functions import mil_loss, contrastive_clustering_loss

model = BotDetectionNet(feature_dim=64, embedding_dim=32)
memory = CrossBatchMemoryBank(feature_dim=32)

scores, features = model(X, mask=mask)
loss = mil_loss(scores, y, k=3, mask=mask) + contrastive_clustering_loss(
    features, scores, y, memory, k=3, mask=mask
)
```

## Observação importante

O método funciona melhor quando o rótulo é fraco por sessão/janela, não necessariamente por requisição. Se o seu `decision` já for por requisição, ainda dá para usar o modelo, mas vale comparar contra uma baseline supervisionada direta.
