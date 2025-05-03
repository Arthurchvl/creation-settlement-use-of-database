DROP SCHEMA "partie1" CASCADE;
CREATE SCHEMA "partie1";
SET SCHEMA 'partie1';

CREATE TABLE _individu (
	id_individu INT,
	nom VARCHAR,
	prenom VARCHAR,
	date_naissance DATE,
	code_postal VARCHAR,
	ville VARCHAR,
	sexe CHAR,
	nationalite VARCHAR,
	INE VARCHAR,
	CONSTRAINT pk_individu PRIMARY KEY (id_individu)
);

CREATE TABLE _etudiant (
	code_nip VARCHAR,
	cat_socio_etu VARCHAR,
	cat_socio_parent VARCHAR,
	bourse_superieur BOOLEAN,
	mention_bac VARCHAR,
	serie_bac VARCHAR,
	dominante_bac VARCHAR,
	specialite_bac VARCHAR,
	mois_annee_obtention_bac VARCHAR(7),
	id_individu INT,
	CONSTRAINT pk_etudiant PRIMARY KEY (code_nip),
	CONSTRAINT uq_individu_etudiant UNIQUE (id_individu),
	CONSTRAINT fk1_etudiant FOREIGN KEY (id_individu) REFERENCES _individu(id_individu)
);

CREATE TABLE _candidat(
	no_candidat INT,
	classement VARCHAR DEFAULT NULL,
	boursier_lycee VARCHAR,
	profil_candidat VARCHAR,
	etablissement VARCHAR,
	dept_etablissement VARCHAR,
	ville_etablissement VARCHAR,
	niveau_etude VARCHAR,
	type_formation VARCHAR,
	serie_prec VARCHAR,
	dominante_prec VARCHAR,
	specialite_prec VARCHAR,
	lv1 VARCHAR,
	lv2 VARCHAR,
	id_individu INT,
	CONSTRAINT pk_candidat PRIMARY KEY (no_candidat),
	CONSTRAINT uq_individu_candidat UNIQUE (id_individu),
	CONSTRAINT fk1_candidat FOREIGN KEY (id_individu) REFERENCES _individu(id_individu)
);

CREATE TABLE _semestre (
	id_semestre INT,
	num_semestre VARCHAR(5),
	annee_univ VARCHAR(9),
	CONSTRAINT pk_smestre PRIMARY KEY (id_semestre)
);

CREATE TABLE _module (
	id_module VARCHAR(5),
	libelle_module VARCHAR,
	ue VARCHAR(2),
	CONSTRAINT pk_module PRIMARY KEY (id_module)
);

CREATE TABLE _programme (
	id_semestre INT,
	id_module VARCHAR(5),
	coefficient NUMERIC,
	CONSTRAINT pk_programme PRIMARY KEY (id_semestre, id_module),
	CONSTRAINT fk1_etudiant FOREIGN KEY (id_semestre) REFERENCES _semestre(id_semestre),
	CONSTRAINT fk2_etudiant FOREIGN KEY (id_module) REFERENCES _module(id_module),
	CONSTRAINT nn_semestre_programme CHECK (id_semestre IS NOT NULL)
);

CREATE TABLE _resultat (
	code_nip VARCHAR,
	id_semestre INT,
	id_module VARCHAR(5),
	moyenne NUMERIC,
	CONSTRAINT pk_resultat PRIMARY KEY (code_nip, id_semestre, id_module),
	CONSTRAINT fk1_resultat FOREIGN KEY (code_nip) REFERENCES _etudiant(code_nip),
	CONSTRAINT fk2_resultat FOREIGN KEY (id_semestre) REFERENCES _semestre(id_semestre),
	CONSTRAINT fk3_resultat FOREIGN KEY (id_module) REFERENCES _module(id_module)
);

CREATE TABLE _inscription(
	code_nip VARCHAR,
	id_semestre INT,
	group_tp VARCHAR(2),
	amenagement_evaluation VARCHAR,
	CONSTRAINT pk_insription PRIMARY KEY (code_nip, id_semestre),
	CONSTRAINT nn_semestre_inscription CHECK (id_semestre IS NOT NULL)
);