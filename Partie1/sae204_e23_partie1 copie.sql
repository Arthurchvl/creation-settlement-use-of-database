DROP SCHEMA IF EXISTS "partie1" CASCADE;

CREATE SCHEMA "partie1";

SET SCHEMA 'partie1';

CREATE TABLE _individu (
	id_individu INT PRIMARY KEY,
	nom VARCHAR,
	prenom VARCHAR,
	date_naissance DATE,
	code_postal VARCHAR,
	ville VARCHAR,
	sexe CHAR,
	nationalite VARCHAR,
	INE VARCHAR
);

CREATE TABLE _etudiant (
	code_nip VARCHAR PRIMARY KEY,
	cat_socio_etu VARCHAR,
	cat_socio_parent VARCHAR,
	bourse_superieur BOOLEAN,
	mention_bac VARCHAR,
	serie_bac VARCHAR,
	dominante_bac VARCHAR,
	specialite_bac VARCHAR,
	mois_annee_obtention_bac VARCHAR(7),
	id_individu INT,
	FOREIGN KEY (id_individu) REFERENCES _individu(id_individu) 
);

CREATE TABLE _semestre (
	id_semestre INT PRIMARY KEY,
	num_semestre VARCHAR(5),
	annee_univ VARCHAR(9)
);

CREATE TABLE _module (
	id_module VARCHAR(5) PRIMARY KEY,
	libelle_module VARCHAR,
	ue VARCHAR(2)
);

CREATE TABLE _inscription(
	group_tp VARCHAR(2),
	amenagement_evaluation VARCHAR,
	code_nip VARCHAR REFERENCES _etudiant(code_nip) ON DELETE CASCADE,
    id_semestre INT REFERENCES _semestre(id_semestre) ON DELETE CASCADE
);