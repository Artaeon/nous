FROM golang:1.23-alpine AS builder

WORKDIR /build
COPY go.mod ./
COPY cmd/ cmd/
COPY internal/ internal/

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o nous ./cmd/nous

FROM alpine:3.20
RUN apk --no-cache add ca-certificates git

WORKDIR /app
COPY --from=builder /build/nous .
COPY packages/ /app/packages/

VOLUME ["/data", "/workspace"]

EXPOSE 3333

ENTRYPOINT ["./nous"]
CMD ["--serve", "--public", "--port", "3333", "--memory", "/data", "--trust"]
