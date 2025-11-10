# API Interface

## OpenAPI Specification

FastAPI automatically creates an OpenAPI spec. You can access it at http://localhost:8000/{version}/docs

The specs can be used to generate interfaces directly, e.g. for Typescript, we recommend [swagger-typescript-api](https://www.npmjs.com/package/swagger-typescript-api).

## API Versioning

The API supports versioning through the URL path. Breaking changes should **always** yield a new version. In order to 
separate versions in OpenAPI specs as well, each version is implemented as its own FastAPI app, which is then mounted by
the main app, adding a prefix.

> [!NOTE]
> Mounting to `/` is not properly working; so as a workaround, we use a router for these endpoints. This is, however, 
> discouraged and you should always create an app and associate it with a given prefix.

### Shared components

Some components might be shared across multiple versions, e.g. models or even controller logic. These should be placed
in the `api.app.shared` package and imported from there. 

> [!CAUTION]
> **Never** make breaking changes to any of the shared types if they interfere with the public API of the versions. For
> example, removing a field from a shared model that is used as a return type **will break all consumers that rely on 
> that field**. Instead, create a new model in the respective version package and use that instead.