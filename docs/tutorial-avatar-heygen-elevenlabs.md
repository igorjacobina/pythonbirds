# Tutorial Prático — Avatar IA com HeyGen + ElevenLabs

> **Objetivo:** gravar o vídeo base da esposa, clonar a voz dela e gerar o **primeiro vídeo prático** (vamos usar a Aula 0.3 — a vitória rápida — como laboratório).
> **Princípio:** acerta o processo UMA vez no Módulo 00. Depois é só repetir pra escalar o curso inteiro.
> ⚠️ Informações conferidas em junho/2026. Interface e preços mudam — confirme no site oficial. Links na seção "Fontes".

---

## Visão geral do fluxo (o caminho completo)

```
1. PREPARAR  →  2. GRAVAR vídeo base (HeyGen)  →  3. GRAVAR áudio limpo (ElevenLabs)
      ↓
4. CLONAR a voz no ElevenLabs  →  5. CRIAR o avatar no HeyGen (+ vídeo de consentimento)
      ↓
6. PLUGAR a voz no HeyGen  →  7. GERAR o vídeo da Aula 0.3  →  8. LEGENDAR e SUBIR na Kiwify
```

**Importante entender:** o vídeo base e o áudio são **duas gravações separadas**, com finalidades diferentes:
- **Vídeo base** = ensina o HeyGen a copiar a IMAGEM/rosto/movimento dela.
- **Áudio limpo** = ensina o ElevenLabs a copiar a VOZ dela.
Não tente usar o áudio do vídeo — grave o áudio à parte, no silêncio, pra ficar limpo.

---

## ETAPA 1 — Preparação (faça bem e o resto fica fácil)

**Equipamento (não precisa ser caro):**
- Celular com câmera boa (ou webcam) + **tripé** (essencial — imagem tremida estraga o avatar).
- **Luz na frente do rosto**, nunca atrás. Luz de janela de dia já resolve. Sem sombra forte no rosto.
- **Ambiente silencioso** pro áudio: quarto com guarda-roupa/cortina/tapete abafa o eco. Desligue ventilador, ar, geladeira por perto, notificações do celular.
- Fone de ouvido com microfone ou um microfone de lapela barato melhora MUITO o áudio (opcional, mas recomendado).

**Aparência da esposa (consistência é tudo):**
- Roupa de cor lisa (evite estampa pequena/listras — "treme" no vídeo).
- Cabelo e maquiagem do jeito que ela quer ser vista no curso INTEIRO (o avatar congela esse visual).
- Fundo neutro e limpo (parede lisa). Pode ser o cenário oficial da marca.

---

## ETAPA 2 — Gravar o VÍDEO BASE (para o avatar no HeyGen)

O HeyGen aprende o rosto e os movimentos dela a partir desse vídeo.

**Especificações:**
- Duração: **2 a 5 minutos** de vídeo.
- Enquadramento: do peito/cintura pra cima, rosto bem visível, centralizado.
- Câmera na **altura dos olhos** (não de baixo).
- Resolução alta (1080p ou mais), celular na horizontal ou vertical conforme o formato do curso.

**Como ela deve se comportar na gravação (o segredo do avatar natural):**
- Olhar **direto pra câmera**, como se falasse com a aluna.
- Falar naturalmente qualquer assunto (pode contar sobre cabelo, do dia, ler um texto) — **o conteúdo não importa**, importa o MOVIMENTO natural: ela gesticulando de leve, sorrindo, mexendo a cabeça como fala normalmente.
- Expressão **leve e simpática**, postura relaxada. Avatar de pessoa "dura" fica robótico.
- Evitar movimentos muito bruscos ou sair do quadro.

> 💡 Dica: grave 2 versões — uma mais séria e uma mais sorridente — e escolha a que gerar o avatar mais natural.

---

## ETAPA 3 — Gravar o ÁUDIO LIMPO (para clonar a voz no ElevenLabs)

Esse é o que mais gente erra. **Áudio limpo = clone bom.** Áudio com eco/ruído = clone ruim.

**Especificações:**
- **Para começar (Clonagem Instantânea):** 1 a 5 minutos de fala limpa (o ideal é 3–5 min).
- **Para qualidade máxima (Clonagem Profissional):** 30 minutos ou mais de gravação de alta qualidade (faremos isso depois de validar — ver Etapa 4).
- Ambiente **mudo**, sem eco, sem barulho de fundo nenhum.
- Ela fala num **tom natural e constante** — o tom que vai usar nas aulas (acolhedor, didático). Como ela fala, o clone vai falar.
- Sem pausas enormes, sem pigarro, sem "é… né…". Pode ler um texto qualquer com naturalidade (até o próprio roteiro da Aula 0.1 serve).
- Salve em boa qualidade (WAV ou MP3 alta taxa).

> 💡 Grave 1 minuto, ouça com fone. Tem chiado? Eco? Refaça o ambiente antes de gravar tudo. 5 min bem gravados valem mais que 30 min ruins.

---

## ETAPA 4 — Clonar a VOZ no ElevenLabs

**Qual plano / tipo usar?**

| Tipo | Precisa de | Plano | Quando usar |
|------|-----------|-------|-------------|
| **Instant Voice Cloning (IVC)** | ~1–5 min de áudio | Starter (~US$6/mês)+ | **Comece por aqui.** Rápido, barato, clone na hora. Perfeito pro laboratório. |
| **Professional Voice Cloning (PVC)** | 30+ min de áudio estúdio | Creator+ | Depois de validar que vende. Clone muito mais fiel, treina em horas. |

**Passo a passo (IVC):**
1. Crie conta no ElevenLabs e assine o plano **Starter** ou superior (IVC não funciona no gratuito, e o pago te dá **direitos comerciais** — essencial pra vender).
2. Vá em **Voices** → **My Voices** → **Add a new voice** → **Instant Voice Clone**.
3. Dê um nome (ex: "Voz [nome da esposa] - Cursos").
4. **Faça upload do áudio limpo** da Etapa 3.
5. **Marque a caixa de consentimento** confirmando que você tem direito de usar essa voz (é a voz da própria esposa, com autorização dela — tudo certo).
6. Confirme. Em segundos o clone está pronto.
7. **Teste:** digite uma frase em português, gere e ouça. Configure o idioma/modelo **multilíngue** pra falar português do Brasil direitinho.
8. **Guarde o "Voice ID"** dessa voz (você vai precisar dele no HeyGen) — fica nas configurações da voz.

> 💡 Dica de pronúncia: pra palavras que a IA erra, escreva foneticamente no roteiro (ex: "babosa" → se errar, ajuste). Pontuação ajuda no ritmo: vírgulas dão pausa, ponto final dá respiro.

---

## ETAPA 5 — Criar o AVATAR no HeyGen (+ vídeo de consentimento)

⚠️ **O HeyGen EXIGE um vídeo de consentimento** antes de criar o avatar. É a trava de segurança deles contra deepfake. Como o avatar é da sua esposa, **ela mesma** grava esse consentimento.

**Passo a passo:**
1. Crie conta no HeyGen. Para uso comercial e sem marca d'água, assine um **plano pago** (confira o plano com direitos comerciais no site).
2. Vá em **Avatars** → **Create Avatar** → escolha o tipo de avatar de vídeo (o que usa a filmagem de 2–5 min; nomes podem variar, ex: "Video Avatar"/"Digital Twin"/"Instant Avatar").
3. **Faça upload do vídeo base** da Etapa 2.
4. **Vídeo de consentimento** (a esposa grava, regras oficiais):
   - Menos de **30 segundos**.
   - Boa luz e áudio claro.
   - Ela lê **exatamente** o texto de consentimento que o HeyGen mostra na tela (não invente — tem que ser o texto deles).
   - Tem que ser **a mesma pessoa** do vídeo do avatar.
   - **Não** filme uma tela mostrando o texto — ela lê e fala olhando pra câmera.
   - Dá pra gravar direto pelo celular: o HeyGen gera um **QR Code**, ela escaneia e envia do próprio telefone. Prático.
5. Envie e aguarde o processamento do avatar (pode levar de minutos a algumas horas).

---

## ETAPA 6 — Plugar a VOZ do ElevenLabs no HeyGen

Você tem **dois caminhos** — escolha um:

**Caminho A — Integrar o ElevenLabs (recomendado, controle total):**
1. No HeyGen, vá em **Settings → Integrations** (ou "Voices → Add voice → third-party").
2. Escolha **ElevenLabs** e **cole sua API Key** do ElevenLabs (pegue em ElevenLabs → perfil → API Keys).
3. Informe o **Voice ID** da voz clonada (da Etapa 4).
4. Pronto — agora a voz dela aparece na lista de vozes do HeyGen.

**Caminho B — Clonar a voz dentro do próprio HeyGen:**
- O HeyGen tem clonagem de voz própria (por baixo, é tecnologia ElevenLabs também). Dá pra clonar lá direto com uma amostra de áudio. Mais simples, menos controle. Serve se você quiser evitar gerenciar duas contas.

> Recomendo o **Caminho A** se você já criou a voz no ElevenLabs — centraliza a voz num lugar só e te dá os recursos avançados do ElevenLabs.

---

## ETAPA 7 — Gerar o PRIMEIRO vídeo (Aula 0.3 como laboratório)

1. No HeyGen, **Create Video** → escolha o formato (vertical pra Reels/tela de celular, ou horizontal).
2. Selecione o **avatar da esposa** (criado na Etapa 5).
3. Selecione a **voz dela** (a do ElevenLabs, da Etapa 6).
4. **Cole o roteiro da Aula 0.3** (arquivo `docs/roteiro-modulo-00.md`) — só a fala, sem as marcações `[ ]`.
   - Use as marcações `[Texto na tela: ...]` como guia pra adicionar **textos/legendas na tela** dentro do editor do HeyGen.
   - Adicione as imagens de apoio (mão aplicando máscara, pente, touca, água fria) como cenas extras.
5. Confira o **idioma** = Português (Brasil).
6. Clique em **Submit/Generate** e aguarde o render.
7. **Assista inteiro com olho crítico** (checklist na próxima seção).

---

## ETAPA 8 — Finalizar e subir na Kiwify

1. **Legendas:** muita aluna assiste sem som. Gere legendas (o HeyGen faz, ou use CapCut/editor) e revise se ficaram certas.
2. **Capa/abertura:** uns 2–3 segundos com o nome da aula.
3. **Exporte** em boa qualidade (1080p).
4. Suba na **área de membros da Kiwify**, no Módulo 00, Aula 0.3.
5. Repita o processo pras outras aulas do Módulo 00.

---

## ✅ Checklist de qualidade (antes de aprovar cada vídeo)

- [ ] O rosto do avatar mexe natural? (sem "boca descolada" do áudio)
- [ ] A voz soa como a esposa e pronuncia o português certo?
- [ ] O áudio está limpo, sem chiado?
- [ ] As legendas estão corretas e sincronizadas?
- [ ] O texto na tela aparece nos momentos certos?
- [ ] A esposa (especialista) revisou e validou TODA recomendação técnica de cabelo?
- [ ] Tem o disclaimer ("conteúdo educativo, não substitui dermatologista")?

---

## 💰 Custo pra rodar o laboratório (estimativa, confira valores atuais)

| Ferramenta | Faixa | Observação |
|-----------|-------|------------|
| ElevenLabs Starter | ~US$ 6/mês | IVC + direitos comerciais. Começa aqui. |
| HeyGen (plano pago) | confira no site | Necessário pra uso comercial e sem marca d'água. |
| Kiwify | grátis pra começar | Cobra taxa só por venda. |

> Dá pra rodar o Módulo 00 inteiro gastando pouco. Só sobe pro plano profissional (PVC no ElevenLabs, plano maior no HeyGen) **depois** que o curso provar que vende.

---

## ⚖️ Consentimento e transparência (não pula isso)

- O avatar e a voz são da **própria esposa, com autorização dela** — então está tudo certo legalmente. Os dois sistemas vão te pedir pra confirmar esse consentimento; é verdade, pode confirmar.
- **Nunca** use rosto/voz de outra pessoa sem autorização — é crime e te quebra o negócio.
- Recomendo deixar claro pra audiência, quando fizer sentido, que o conteúdo é apresentado com apoio de IA. Transparência protege a marca a longo prazo.

---

## Fontes (conferidas em junho/2026)
- HeyGen — Recording your Consent Video: https://help.heygen.com/en/articles/12092609-recording-your-consent-video
- HeyGen — Digital Twin / Video Avatar FAQ: https://help.heygen.com/en/articles/9380615-video-avatar-faq
- HeyGen — Integrar ElevenLabs e vozes de terceiros: https://help.heygen.com/en/articles/8310663-how-to-integrate-elevenlabs-other-third-party-voices
- ElevenLabs — Voice Cloning (overview): https://elevenlabs.io/docs/creative-platform/voices/voice-cloning
- ElevenLabs — Voice cloning, como funciona: https://elevenlabs.io/docs/eleven-api/concepts/voice-cloning

---

## Próximo passo
Depois que o primeiro vídeo da Aula 0.3 sair bom, a gente:
1. Gera as outras 3 aulas do Módulo 00 (mesmo processo).
2. Configura o produto + checkout + garantia de 7 dias na Kiwify.
3. Escreve o roteiro do **Módulo 02 (Queda e Quebra)** — o que mais vende.
